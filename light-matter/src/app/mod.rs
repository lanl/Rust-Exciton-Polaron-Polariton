use exciton_2d::ExcitonConfig;
use matrix_ops::{ComposedOperator, MatrixOps, Result};
use ndarray::{Array1, Array2, arr1};
use ndarray_linalg::c64;
use sprs::CsMat;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use crate::cli;
use crate::visualization::{DielectricPlotType, plot_dielectric_function};
use crate::{band_structure, dielectric_function, hamiltonian, visualization};
use crate::{timed_println, timed_println_rank0};

use crate::cli::{BandParallelModeArg, Cli, Command, MethodArg};

use crate::config::PPConfig;

use crate::visualization_dielectric::generate_stacked_plots;

use crate::hamiltonian::{PPHamiltonian, SingExBosons};

pub enum Method {
    Sparse,
    Dense,
    LinOp,
}

fn run_bandstructure_with_args(
    method: Method,
    e_config: ExcitonConfig,
    args: cli::BandstructureArgs,
    chunk_index: usize,
    n_chunks: usize,
) -> Result<()> {
    let a0 = e_config.a0;
    let v0 = e_config.v0;
    let n_eig = args.n_eig;
    let n_k_points = args.n_k_points;
    let zoomed = args.zoomed;
    let total_coupling = args.total_coupling;

    timed_println_rank0!("total_coupling = {}", total_coupling);

    let (pp_config, sym_points, sym_labels) = build_bandstructure_pp_config(e_config, &args);

    let n_modes = pp_config.q_values.len();
    let coupling = if n_modes == 0 {
        0.0
    } else {
        pp_config.lm_couplings.iter().sum::<f64>() / n_modes as f64
    };

    timed_println_rank0!("Running with {} photon modes", n_modes);

    match method {
        Method::Sparse => run_bandstructure_for_backend::<CsMat<c64>>(
            "sparse",
            &args,
            &pp_config,
            &sym_points,
            &sym_labels,
            n_k_points,
            n_eig,
            a0,
            v0,
            total_coupling,
            n_modes,
            coupling,
            zoomed,
            chunk_index,
            n_chunks,
        ),

        Method::Dense => run_bandstructure_for_backend::<Array2<c64>>(
            "dense",
            &args,
            &pp_config,
            &sym_points,
            &sym_labels,
            n_k_points,
            n_eig,
            a0,
            v0,
            total_coupling,
            n_modes,
            coupling,
            zoomed,
            chunk_index,
            n_chunks,
        ),

        Method::LinOp => run_bandstructure_for_backend::<ComposedOperator<c64>>(
            "linear-operator",
            &args,
            &pp_config,
            &sym_points,
            &sym_labels,
            n_k_points,
            n_eig,
            a0,
            v0,
            total_coupling,
            n_modes,
            coupling,
            zoomed,
            chunk_index,
            n_chunks,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_bandstructure_for_backend<M>(
    backend_name: &str,
    args: &cli::BandstructureArgs,
    pp_config: &PPConfig,
    sym_points: &Vec<Vec<f64>>,
    sym_labels: &Vec<String>,
    n_k_points: usize,
    n_eig: usize,
    a0: f64,
    v0: f64,
    total_coupling: f64,
    n_modes: usize,
    coupling: f64,
    zoomed: bool,
    chunk_index: usize,
    n_chunks: usize,
) -> Result<()>
where
    M: MatrixOps + matrix_ops::Diagonalize,
    M::Scalar: matrix_ops::Sqrt
        + std::iter::Sum
        + matrix_ops::ConvertFrom<f64>
        + matrix_ops::ConvertFrom<c64>,
    SingExBosons<M>: hamiltonian::Boson<M>,
{
    timed_println_rank0!("Using {} backend for light-matter coupling.", backend_name);

    let start_time = Instant::now();

    let mut pp: PPHamiltonian<M, SingExBosons<M>> = PPHamiltonian::from_config(pp_config);

    let (k_points_color, eigvals_list_color, photonic_character_list);

    if args.calculate {
        let time1 = Instant::now();

        timed_println_rank0!("Calculating {} eigenstates", n_eig);
        timed_println_rank0!("Calculating color-coded band structure...");

        (k_points_color, eigvals_list_color, photonic_character_list) =
            run_bandstructure_parallel_mode(
                &mut pp,
                args.parallel,
                n_k_points,
                n_eig,
                sym_points,
                pp_config,
                args.use_cached,
                zoomed,
                chunk_index,
                n_chunks,
            )?;

        timed_println!(
            "Calculated band structure in {:?}",
            Instant::now().duration_since(time1)
        );
    } else {
        (k_points_color, eigvals_list_color, photonic_character_list) =
            band_structure::load_color_band_structure(&pp, n_k_points, n_eig, pp_config, zoomed);
    }

    timed_println!("Calc dispersion in {:?}", start_time.elapsed());

    let mut save_path = format!("sing_ex_color_disp_plot_{}_n{}", total_coupling, n_modes);
    save_path += if zoomed { "_zoomed.pdf" } else { ".pdf" };

    if !k_points_color.is_empty()
        && sim_common::mpi::is_mpi_rank0_env()
        && !matches!(args.parallel, BandParallelModeArg::DistributedChunks)
    {
        if let Err(e) = band_structure::plot_color_band_structure(
            &k_points_color,
            sym_points,
            sym_labels,
            &eigvals_list_color,
            &photonic_character_list,
            n_eig,
            a0,
            &save_path,
            v0,
            coupling,
            zoomed,
        ) {
            timed_println!("Error plotting color-coded band structure: {}", e);
        }
    }

    Ok(())
}

fn run_bandstructure_parallel_mode<M, B>(
    pp: &mut PPHamiltonian<M, B>,
    parallel: BandParallelModeArg,
    n_k_points: usize,
    n_eig: usize,
    sym_points: &Vec<Vec<f64>>,
    pp_config: &PPConfig,
    use_cached: bool,
    zoomed: bool,
    chunk_index: usize,
    n_chunks: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>)>
where
    M: MatrixOps + matrix_ops::Diagonalize,
    M::Scalar: matrix_ops::Sqrt
        + std::iter::Sum
        + matrix_ops::ConvertFrom<f64>
        + matrix_ops::ConvertFrom<c64>,
    B: hamiltonian::Boson<M>,
{
    match parallel {
        BandParallelModeArg::Serial => {
            // Serial is equivalent to a single distributed chunk.
            band_structure::calculate_color_band_structure_distributed_chunk(
                pp, n_k_points, n_eig, sym_points, pp_config, use_cached, zoomed, 0, 1,
            )
        }
        BandParallelModeArg::DistributedChunks => {
            band_structure::calculate_color_band_structure_distributed_chunk(
                pp,
                n_k_points,
                n_eig,
                sym_points,
                pp_config,
                use_cached,
                zoomed,
                chunk_index,
                n_chunks,
            )
        }

        BandParallelModeArg::GatherChunks => band_structure::gather_color_band_structure_chunks(
            n_k_points, n_eig, pp_config, zoomed, n_chunks,
        ),
    }
}

/// Main function demonstrating the use of the light-matter coupling framework
/// Calculate dielectric function for a range of coupling strengths
fn run_dielectric_scan(e_config: ExcitonConfig, args: cli::DielectricArgs) -> Result<()> {
    timed_println_rank0!("Starting dielectric function scan...");

    let calculate = args.calculate;
    let use_cached = args.use_cached;
    let zoomed = args.zoomed;

    let n_rel = args.n_rel;
    let n_ex_eig = args.n_ex_eig;

    let omega_0 = args.omega_0;
    let n_modes = args.n_modes;

    let omega_max = args.omega_max;
    let reflectivities = args.reflectivities.clone();
    let d_omega = args.d_omega;

    let pn_coupling = args.pn_coupling;
    let pn_omega = args.pn_omega;
    let pn_k = arr1(&[args.pn_k_x, 0.0]);
    let pn_n_fock = args.pn_n_fock;

    let n_eig = args.n_eig;
    let omegas = Array1::linspace(
        args.omega_min_scan,
        args.omega_max_scan,
        args.n_omega_points,
    );

    let gamma = args.gamma.unwrap_or(d_omega);

    let omega_range = match (args.plot_omega_min, args.plot_omega_max) {
        (Some(min), Some(max)) => Some((min, max)),
        _ => {
            if !zoomed {
                Some((0.182, 0.202))
            } else {
                Some((0.183, 0.188))
            }
        }
    };

    let save_dir = args.output_dir.as_str();
    std::fs::create_dir_all(save_dir).unwrap_or_else(|_| {
        timed_println_rank0!(
            "Directory {} already exists or couldn't be created",
            save_dir
        );
    });

    timed_println_rank0!("Dielectric CLI configuration:");
    timed_println_rank0!("  calculate       = {}", calculate);
    timed_println_rank0!("  use_cached      = {}", use_cached);
    timed_println_rank0!("  zoomed          = {}", zoomed);
    timed_println_rank0!("  n_rel           = {}", n_rel);
    timed_println_rank0!("  n_ex_eig        = {}", n_ex_eig);
    timed_println_rank0!("  n_eig           = {}", n_eig);
    timed_println_rank0!("  omega_0         = {}", omega_0);
    timed_println_rank0!("  omega_max       = {}", omega_max);
    timed_println_rank0!("  d_omega         = {}", d_omega);
    timed_println_rank0!("  gamma           = {}", gamma);
    timed_println_rank0!("  n_modes target  = {}", n_modes);
    timed_println_rank0!("  reflectivities  = {:?}", reflectivities);
    timed_println_rank0!("  pn_coupling     = {}", pn_coupling);
    timed_println_rank0!("  pn_omega        = {}", pn_omega);
    timed_println_rank0!("  pn_k            = {:?}", pn_k);
    timed_println_rank0!("  pn_n_fock       = {}", pn_n_fock);
    timed_println_rank0!(
        "  omega scan      = [{}, {}], n={}",
        args.omega_min_scan,
        args.omega_max_scan,
        args.n_omega_points
    );
    timed_println_rank0!("  output_dir      = {}", save_dir);

    // Generate photon modes and couplings for each reflectivity/mode-construction choice.
    let mut all_q_values = Vec::new();
    let mut all_mode_couplings = Vec::new();
    let mut total_coupling_values = Vec::new();

    for &reflectivity in reflectivities.iter() {
        let mode_set = crate::modes::photon::from_dielectric_args(&args, reflectivity);
        let total_coupling: f64 = mode_set.couplings.iter().sum();

        timed_println_rank0!(
            "Reflectivity {}: generated {} modes with total coupling {:.6} using {:?}",
            reflectivity,
            mode_set.q_values.len(),
            total_coupling,
            args.photon_modes
        );

        all_q_values.push(mode_set.q_values);
        all_mode_couplings.push(mode_set.couplings);
        total_coupling_values.push(total_coupling);
    }

    let total_coupling_values = Array1::from_vec(total_coupling_values);

    // Validate mode counts.
    if let Some(first) = all_q_values.first() {
        let n_modes_first = first.len();

        for (idx, q_vals) in all_q_values.iter().enumerate() {
            if q_vals.len() != n_modes_first {
                timed_println_rank0!(
                    "Warning: Reflectivity {} generated {} modes, but first generated {}.",
                    reflectivities[idx],
                    q_vals.len(),
                    n_modes_first
                );
            }
        }
    }

    // Output plot filenames.
    let suffix = if zoomed { "_zoomed.svg" } else { ".svg" };

    let real_filename = format!(
        "{}/dielectric_real_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}{}",
        save_dir, e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock, suffix,
    );

    let imag_filename = format!(
        "{}/dielectric_imag_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}{}",
        save_dir, e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock, suffix,
    );

    let both_filename = format!(
        "{}/dielectric_both_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}{}",
        save_dir, e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock, suffix,
    );

    if !calculate {
        if !sim_common::mpi::is_mpi_rank0_env() {
            return Ok(());
        }

        timed_println!("Plotting dielectric function from existing data...");

        let q_values = &all_q_values[0];

        let ref_params = visualization::DielectricParams {
            a0: e_config.a0,
            v0: e_config.v0,
            n_rel,
            lm_coupling: 0.0,
            n_modes: q_values.len(),
            pn_coupling,
            pn_n_fock,
            n_eig,
            gamma,
        };

        match plot_dielectric_function(
            save_dir,
            Some(&real_filename),
            DielectricPlotType::Real,
            omega_range,
            &total_coupling_values,
            Some(&ref_params),
            zoomed,
            Some(q_values),
        ) {
            Ok(_) => timed_println!("Successfully plotted real part of dielectric function"),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }

        match plot_dielectric_function(
            save_dir,
            Some(&imag_filename),
            DielectricPlotType::Imaginary,
            omega_range,
            &total_coupling_values,
            Some(&ref_params),
            zoomed,
            None,
        ) {
            Ok(_) => timed_println!("Successfully plotted imaginary part of dielectric function"),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }

        match plot_dielectric_function(
            save_dir,
            Some(&both_filename),
            DielectricPlotType::Both,
            omega_range,
            &total_coupling_values,
            Some(&ref_params),
            zoomed,
            Some(q_values),
        ) {
            Ok(_) => timed_println!("Successfully plotted both parts of dielectric function"),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }

        generate_stacked_plots(
            &total_coupling_values,
            &all_mode_couplings,
            &all_q_values,
            &reflectivities,
            &e_config,
            n_rel,
            n_ex_eig,
            omega_0,
            d_omega,
            omega_max,
            pn_coupling,
            pn_omega,
            &pn_k,
            pn_n_fock,
            n_eig,
            &omegas,
            gamma,
            save_dir,
            zoomed,
            omega_range,
        );

        return Ok(());
    }

    timed_println_rank0!(
        "Starting dielectric function scan for {} reflectivity values",
        reflectivities.len()
    );

    // Select exactly one reflectivity for this process group.
    // PETSc/SLEPc solves are collective over COMM_WORLD, so use job arrays for scan parallelism.
    let refl_idx: usize = args
        .reflectivity_index
        .or_else(|| sim_common::mpi::env_usize(&["REFLECTIVITY_INDEX", "SLURM_ARRAY_TASK_ID"]))
        .unwrap_or(0);

    if refl_idx >= reflectivities.len() {
        panic!(
            "Reflectivity index {} out of range 0..{}",
            refl_idx,
            reflectivities.len()
        );
    }

    let total_coupling = total_coupling_values[refl_idx];
    let q_values = &all_q_values[refl_idx];
    let reflectivity = reflectivities[refl_idx];

    timed_println_rank0!(
        "Selected reflectivity index {}: R={}, modes={}, total coupling={:.6}",
        refl_idx,
        reflectivity,
        q_values.len(),
        total_coupling
    );

    let start_time = Instant::now();

    let pp_config = build_dielectric_pp_config_for_reflectivity(e_config.clone(), &args, refl_idx);

    timed_println_rank0!("Using ComposedOperator + distributed SLEPc shell backend");

    let mut pp: PPHamiltonian<ComposedOperator<c64>, SingExBosons<ComposedOperator<c64>>> =
        PPHamiltonian::from_config(&pp_config);

    let result = dielectric_function::get_dielectric_function(
        &pp_config,
        &mut pp,
        n_eig,
        omegas.clone(),
        gamma,
        save_dir,
        use_cached,
    )?;

    let filename = dielectric_function::get_dielectric_function_filename(
        &pp, &pp_config, save_dir, n_eig, gamma,
    );

    if sim_common::mpi::is_mpi_rank0_env() {
        timed_println!("Saving dielectric function data to {}", filename);

        let mut file = File::create(&filename).unwrap();

        writeln!(file, "# omega Re[eps] Im[eps]").unwrap();

        for i in 0..omegas.len() {
            writeln!(
                file,
                "{:.6} {:.6} {:.6}",
                omegas[i], result.real[i], result.imag[i]
            )
            .unwrap();
        }
    }

    timed_println_rank0!(
        "Completed reflectivity index {} in {:?}",
        refl_idx,
        start_time.elapsed()
    );

    Ok(())
}

fn build_bandstructure_pp_config(
    e_config: ExcitonConfig,
    args: &cli::BandstructureArgs,
) -> (PPConfig, Vec<Vec<f64>>, Vec<String>) {
    let a0 = e_config.a0;

    let mode_set = crate::modes::photon::from_bandstructure_args(args);

    let pn_k = arr1(&[0.0, args.pn_k_y]);

    let pp_config = match mode_set.kind {
        crate::modes::photon::PhotonModeKind::Effective {
            d_omega,
            omega_max,
            reflectivity,
            ..
        } => PPConfig::new_with_effective_modes(
            e_config,
            args.n_rel,
            args.n_ex_eig,
            mode_set.couplings,
            args.omega_0,
            mode_set.q_values,
            args.pn_coupling,
            args.pn_omega,
            pn_k,
            args.pn_n_fock,
            args.n_k_points,
            args.n_bands,
            d_omega,
            omega_max,
            reflectivity,
        ),
        crate::modes::photon::PhotonModeKind::Cartesian { .. } => PPConfig::new(
            e_config,
            args.n_rel,
            args.n_ex_eig,
            mode_set.couplings,
            args.omega_0,
            mode_set.q_values,
            args.pn_coupling,
            args.pn_omega,
            pn_k,
            args.pn_n_fock,
            args.n_k_points,
            args.n_bands,
        ),
    };

    let (sym_points, sym_labels) = crate::modes::photon::square_lattice_symmetry_points(a0);

    (pp_config, sym_points, sym_labels)
}

fn selected_dielectric_reflectivity_index(args: &cli::DielectricArgs) -> usize {
    args.reflectivity_index
        .or_else(|| sim_common::mpi::env_usize(&["REFLECTIVITY_INDEX", "SLURM_ARRAY_TASK_ID"]))
        .unwrap_or(0)
}

fn build_dielectric_pp_config_for_reflectivity(
    e_config: ExcitonConfig,
    args: &cli::DielectricArgs,
    refl_idx: usize,
) -> PPConfig {
    if refl_idx >= args.reflectivities.len() {
        panic!(
            "Reflectivity index {} out of range 0..{}",
            refl_idx,
            args.reflectivities.len()
        );
    }

    let reflectivity = args.reflectivities[refl_idx];
    let mode_set = crate::modes::photon::from_dielectric_args(args, reflectivity);

    let pn_k = arr1(&[args.pn_k_x, 0.0]);

    match mode_set.kind {
        crate::modes::photon::PhotonModeKind::Effective {
            d_omega,
            omega_max,
            reflectivity,
            ..
        } => PPConfig::new_with_effective_modes(
            e_config,
            args.n_rel,
            args.n_ex_eig,
            mode_set.couplings,
            args.omega_0,
            mode_set.q_values,
            args.pn_coupling,
            args.pn_omega,
            pn_k,
            args.pn_n_fock,
            1,
            1,
            d_omega,
            omega_max,
            reflectivity,
        ),
        crate::modes::photon::PhotonModeKind::Cartesian { .. } => PPConfig::new(
            e_config,
            args.n_rel,
            args.n_ex_eig,
            mode_set.couplings,
            args.omega_0,
            mode_set.q_values,
            args.pn_coupling,
            args.pn_omega,
            pn_k,
            args.pn_n_fock,
            1,
            1,
        ),
    }
}

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

    let k_points = band_structure::generate_band_k_points(
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

fn run_matrix_elem_calc_cli(args: cli::MatrixElemCalcArgs) -> Result<()> {
    match args.target {
        cli::MatrixElemCalcTarget::Bandstructure(args) => {
            run_matrix_elem_calc_bandstructure_cli(args)
        }
        cli::MatrixElemCalcTarget::Dielectric(args) => run_matrix_elem_calc_dielectric_cli(args),
    }
}

fn method_from_arg(method: MethodArg) -> Method {
    match method {
        MethodArg::Sparse => Method::Sparse,
        MethodArg::Dense => Method::Dense,
        MethodArg::Linop => Method::LinOp,
    }
}

fn run_bandstructure_cli(args: cli::BandstructureArgs) -> Result<()> {
    let e_config = args.exciton.to_config();
    let method = method_from_arg(args.method);

    let chunk_index = args
        .k_chunk_index
        .or_else(|| sim_common::mpi::env_usize(&["K_CHUNK_INDEX", "SLURM_ARRAY_TASK_ID"]))
        .unwrap_or(0);

    let n_chunks = args
        .k_n_chunks
        .or_else(|| sim_common::mpi::env_usize(&["K_N_CHUNKS", "SLURM_ARRAY_TASK_COUNT"]))
        .unwrap_or(1);

    run_bandstructure_with_args(method, e_config, args, chunk_index, n_chunks)
}

fn run_dielectric_cli(args: cli::DielectricArgs) -> Result<()> {
    let e_config = args.exciton.to_config();
    run_dielectric_scan(e_config, args)
}

pub fn run(cli: Cli) -> Result<()> {
    let num_threads = cli
        .rayon_threads
        .or_else(|| {
            std::env::var("OMP_NUM_THREADS")
                .ok()
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(1);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    match cli.command {
        Command::Bandstructure(args) => run_bandstructure_cli(args)?,
        Command::Dielectric(args) => run_dielectric_cli(args)?,
        Command::MatrixElemCalc(args) => run_matrix_elem_calc_cli(args)?,
    }

    Ok(())
}
