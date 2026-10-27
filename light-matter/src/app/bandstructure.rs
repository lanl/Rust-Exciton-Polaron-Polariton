use exciton_2d::ExcitonConfig;
use matrix_ops::{ComposedOperator, MatrixOps, MatrixOpsError, Result};
use ndarray::{Array1, Array2};
use ndarray_linalg::c64;
use sprs::CsMat;
use std::time::Instant;

use super::config::{Method, build_bandstructure_pp_config, method_from_arg};
use crate::cli;
use crate::cli::BandParallelModeArg;
use crate::config::PPConfig;
use crate::hamiltonian::{self, PPHamiltonian, SingExBosons};
use crate::utils::eig_selection_tag;
use crate::{bands, timed_println, timed_println_rank0};

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
    let energy_cutoff = args.energy_cutoff;
    let n_k_points = args.n_k_points;
    let zoomed = args.zoomed;
    let total_coupling = args.total_coupling;

    if let Some(cutoff) = energy_cutoff {
        if !cutoff.is_finite() {
            return Err(MatrixOpsError::Other {
                message: format!("--energy-cutoff must be finite, got {}", cutoff),
            });
        }
    }

    timed_println_rank0!("total_coupling = {}", total_coupling);

    match energy_cutoff {
        Some(cutoff) => timed_println_rank0!(
            "energy_cutoff = {:.6e}: ALL states with E <= cutoff per K (n_eig = {} is only a batch hint); \
             plot y-max = cutoff",
            cutoff,
            n_eig
        ),
        None => timed_println_rank0!("energy_cutoff = none: lowest n_eig = {} states per K", n_eig),
    }

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
            energy_cutoff,
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
            energy_cutoff,
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
            energy_cutoff,
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

fn run_bandstructure_for_backend<M>(
    backend_name: &str,
    args: &cli::BandstructureArgs,
    pp_config: &PPConfig,
    sym_points: &Vec<Vec<f64>>,
    sym_labels: &Vec<String>,
    n_k_points: usize,
    n_eig: usize,
    energy_cutoff: Option<f64>,
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

    let (k_points_color, eigvals_list_color, photonic_character_list);

    if args.calculate {
        let time1 = Instant::now();

        let mut pp: PPHamiltonian<M, SingExBosons<M>> = PPHamiltonian::from_config(pp_config);

        match energy_cutoff {
            Some(cutoff) => timed_println_rank0!(
                "Calculating all eigenstates below E = {:.6e} at each K point",
                cutoff
            ),
            None => timed_println_rank0!("Calculating {} eigenstates at each K point", n_eig),
        }
        timed_println_rank0!("Calculating color-coded band structure...");

        let checkpoint = args.ckpt.to_checkpoint();
        if checkpoint.is_some() && energy_cutoff.is_none() {
            timed_println_rank0!(
                "Warning: --checkpoint only applies to --energy-cutoff solves; ignoring"
            );
        }

        (k_points_color, eigvals_list_color, photonic_character_list) =
            run_bandstructure_parallel_mode(
                &mut pp,
                args.parallel,
                n_k_points,
                n_eig,
                energy_cutoff,
                sym_points,
                pp_config,
                args.use_cached,
                zoomed,
                chunk_index,
                n_chunks,
                checkpoint.as_ref(),
            )?;

        timed_println!(
            "Calculated band structure in {:?}",
            Instant::now().duration_since(time1)
        );
    } else {
        if !sim_common::mpi::is_mpi_rank0_env() {
            return Ok(());
        }

        (k_points_color, eigvals_list_color, photonic_character_list) =
            bands::load_color_band_structure(pp_config, n_k_points, n_eig, energy_cutoff, zoomed)?;
    }

    timed_println!("Calc dispersion in {:?}", start_time.elapsed());

    let mut save_path = format!("sing_ex_color_disp_plot_{}_n{}", total_coupling, n_modes);
    if energy_cutoff.is_some() {
        save_path += &format!("_{}", eig_selection_tag(n_eig, energy_cutoff));
    }
    save_path += if zoomed { "_zoomed.svg" } else { ".svg" };

    let skip_plot_for_chunk_job =
        args.calculate && matches!(args.parallel, BandParallelModeArg::DistributedChunks);

    if !k_points_color.is_empty() && sim_common::mpi::is_mpi_rank0_env() && !skip_plot_for_chunk_job
    {
        if let Err(e) = bands::plot_color_band_structure(
            &k_points_color,
            sym_points,
            sym_labels,
            &eigvals_list_color,
            &photonic_character_list,
            n_eig,
            energy_cutoff,
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
    energy_cutoff: Option<f64>,
    sym_points: &Vec<Vec<f64>>,
    pp_config: &PPConfig,
    use_cached: bool,
    zoomed: bool,
    chunk_index: usize,
    n_chunks: usize,
    checkpoint: Option<&crate::checkpoint::DiagCheckpoint>,
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
        BandParallelModeArg::Serial => bands::calculate_color_band_structure_distributed_chunk(
            pp, n_k_points, n_eig, energy_cutoff, sym_points, pp_config, use_cached, zoomed,
            0, 1, checkpoint,
        ),
        BandParallelModeArg::DistributedChunks => {
            bands::calculate_color_band_structure_distributed_chunk(
                pp, n_k_points, n_eig, energy_cutoff, sym_points, pp_config, use_cached, zoomed,
                chunk_index, n_chunks, checkpoint,
            )
        }
        BandParallelModeArg::GatherChunks => bands::gather_color_band_structure_chunks(
            n_k_points, n_eig, energy_cutoff, pp_config, zoomed, n_chunks,
        ),
    }
}

pub(super) fn run_bandstructure_cli(args: cli::BandstructureArgs) -> Result<()> {
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