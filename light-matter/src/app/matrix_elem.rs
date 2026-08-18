use matrix_ops::Result;
use ndarray_linalg::c64;
use sprs::CsMat;

use super::config::{
    build_bandstructure_pp_config, build_dielectric_pp_config_for_reflectivity,
    selected_dielectric_reflectivity_index,
};
use crate::cli;
use crate::hamiltonian::{PPHamiltonian, SingExBosons};
use crate::{bands, timed_println};

fn matrix_elem_band_k_indices(
    args: &cli::MatrixElemBandstructureArgs,
    n_k_points_total: usize,
) -> Vec<usize> {
    if args.all_k_points {
        return (0..n_k_points_total).collect();
    }

    // Explicit single-K selection.
    if let Some(k_index) = args
        .k_index
        .or_else(|| sim_common::mpi::env_usize(&["K_INDEX"]))
    {
        if k_index >= n_k_points_total {
            panic!("K index {} out of range 0..{}", k_index, n_k_points_total);
        }

        return vec![k_index];
    }

    // Chunked precompute selection.
    let n_chunks = args
        .bandstructure
        .k_n_chunks
        .or_else(|| sim_common::mpi::env_usize(&["K_N_CHUNKS"]));

    if let Some(n_chunks) = n_chunks {
        let chunk_index = args
            .bandstructure
            .k_chunk_index
            .or_else(|| sim_common::mpi::env_usize(&["K_CHUNK_INDEX", "SLURM_ARRAY_TASK_ID"]))
            .unwrap_or(0);

        let (start, end) = sim_common::mpi::chunk_bounds(n_k_points_total, chunk_index, n_chunks);

        return (start..end).collect();
    }

    // Common Slurm-array case: one array task per K point.
    if let Some(k_index) = sim_common::mpi::env_usize(&["SLURM_ARRAY_TASK_ID"]) {
        if k_index >= n_k_points_total {
            panic!(
                "SLURM_ARRAY_TASK_ID={} out of K-index range 0..{}",
                k_index, n_k_points_total
            );
        }

        return vec![k_index];
    }

    // Safe local default.
    vec![0]
}

fn run_matrix_elem_calc_bandstructure_cli(args: cli::MatrixElemBandstructureArgs) -> Result<()> {
    // Matrix-element precompute should generally run as one process with many
    // Rayon threads. Avoid many MPI ranks writing the same cache files.
    if !sim_common::mpi::is_mpi_rank0_env() {
        return Ok(());
    }

    let e_config = args.bandstructure.exciton.to_config();

    let (mut pp_config, sym_points, _sym_labels) =
        build_bandstructure_pp_config(e_config, &args.bandstructure);

    pp_config.cache_dir = Some(args.common.cache_dir.clone());

    std::fs::create_dir_all(&args.common.cache_dir).unwrap_or_else(|e| {
        panic!(
            "Failed to create cache directory {}: {}",
            args.common.cache_dir, e
        );
    });

    let k_points = bands::generate_band_k_points(
        args.bandstructure.n_k_points,
        &sym_points,
        pp_config.e_config.a0,
        args.bandstructure.zoomed,
    );

    let k_indices = matrix_elem_band_k_indices(&args, k_points.len());

    timed_println!("Starting bandstructure LM matrix-element precompute");
    timed_println!("  n_k_points        = {}", k_points.len());
    timed_println!("  selected K inds   = {:?}", k_indices);
    timed_println!("  n_rel             = {}", args.bandstructure.n_rel);
    timed_println!("  n_ex_eig          = {}", args.bandstructure.n_ex_eig);
    timed_println!("  omega_0           = {}", args.bandstructure.omega_0);
    timed_println!("  n_modes side      = {}", args.bandstructure.n_modes);
    timed_println!("  photon modes      = {}", pp_config.q_values.len());
    timed_println!("  q_max             = {}", args.bandstructure.q_max);
    timed_println!(
        "  total coupling    = {}",
        args.bandstructure.total_coupling
    );
    timed_println!("  cache_dir         = {}", args.common.cache_dir);
    timed_println!("  force_recompute   = {}", args.common.force_recompute);
    timed_println!("  base PP hash      = {:016x}", pp_config.get_config_hash());

    for &k_index in &k_indices {
        let k_point = k_points[k_index].clone();

        timed_println!(
            "Precomputing matrix elements for K index {} / {}: {:?}",
            k_index,
            k_points.len(),
            k_point
        );

        timed_println!(
            "  quantized K       = {:?}, quantum={}",
            k_point
                .iter()
                .map(|&x| crate::utils::quantize_f64_for_hash(x))
                .collect::<Vec<_>>(),
            crate::utils::K_HASH_QUANTUM,
        );

        let mut pp_config_k = pp_config.clone();
        pp_config_k.e_config.k = k_point.clone();

        let mut pp: PPHamiltonian<CsMat<c64>, SingExBosons<CsMat<c64>>> =
            PPHamiltonian::from_config(&pp_config_k);

        pp.polariton
            .compute_lm_matrix_elements_only(Some(k_point), args.common.force_recompute)?;

        drop(pp);

        timed_println!(
            "Finished and dropped PPHamiltonian for K index {} / {}",
            k_index,
            k_points.len()
        );
    }

    timed_println!("Bandstructure LM matrix-element precompute finished");

    Ok(())
}

fn run_matrix_elem_calc_dielectric_cli(args: cli::MatrixElemDielectricArgs) -> Result<()> {
    // Matrix-element precompute should generally run as one process with many
    // Rayon threads. Avoid many MPI ranks writing the same cache files.
    if !sim_common::mpi::is_mpi_rank0_env() {
        return Ok(());
    }

    let mut e_config = args.dielectric.exciton.to_config();

    // Dielectric calculation is currently a Gamma-point calculation.
    let k_point = vec![0.0, 0.0];
    e_config.k = k_point.clone();

    let refl_idx = selected_dielectric_reflectivity_index(&args.dielectric);

    let mut pp_config =
        build_dielectric_pp_config_for_reflectivity(e_config, &args.dielectric, refl_idx);

    pp_config.cache_dir = Some(args.common.cache_dir.clone());

    std::fs::create_dir_all(&args.common.cache_dir).unwrap_or_else(|e| {
        panic!(
            "Failed to create cache directory {}: {}",
            args.common.cache_dir, e
        );
    });

    let reflectivity = args.dielectric.reflectivities[refl_idx];
    let total_coupling: f64 = pp_config.lm_couplings.iter().sum();

    timed_println!("Starting dielectric LM matrix-element precompute");
    timed_println!("  k point           = {:?}", k_point);
    timed_println!(
        "  quantized K       = {:?}, quantum={}",
        k_point
            .iter()
            .map(|&x| crate::utils::quantize_f64_for_hash(x))
            .collect::<Vec<_>>(),
        crate::utils::K_HASH_QUANTUM,
    );
    timed_println!("  reflectivity idx  = {}", refl_idx);
    timed_println!("  reflectivity      = {}", reflectivity);
    timed_println!("  n_rel             = {}", args.dielectric.n_rel);
    timed_println!("  n_ex_eig          = {}", args.dielectric.n_ex_eig);
    timed_println!("  omega_0           = {}", args.dielectric.omega_0);
    timed_println!("  omega_max         = {}", args.dielectric.omega_max);
    timed_println!("  d_omega           = {}", args.dielectric.d_omega);
    timed_println!("  n_modes target    = {}", args.dielectric.n_modes);
    timed_println!("  photon modes      = {}", pp_config.q_values.len());
    timed_println!("  total coupling    = {:.8}", total_coupling);
    timed_println!("  cache_dir         = {}", args.common.cache_dir);
    timed_println!("  force_recompute   = {}", args.common.force_recompute);
    timed_println!("  base PP hash      = {:016x}", pp_config.get_config_hash());

    let mut pp: PPHamiltonian<CsMat<c64>, SingExBosons<CsMat<c64>>> =
        PPHamiltonian::from_config(&pp_config);

    pp.polariton
        .compute_lm_matrix_elements_only(Some(k_point), args.common.force_recompute)?;

    timed_println!("Dielectric LM matrix-element precompute finished");

    Ok(())
}

pub(super) fn run_matrix_elem_calc_cli(args: cli::MatrixElemCalcArgs) -> Result<()> {
    match args.target {
        cli::MatrixElemCalcTarget::Bandstructure(args) => {
            run_matrix_elem_calc_bandstructure_cli(args)
        }
        cli::MatrixElemCalcTarget::Dielectric(args) => run_matrix_elem_calc_dielectric_cli(args),
    }
}
