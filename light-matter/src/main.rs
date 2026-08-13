use exciton_2d::ExcitonConfig;
use matrix_ops::{ComposedOperator, Result, MatrixOps};
use ndarray::{Array1, Array2, arr1};
use ndarray_linalg::c64;
use sprs::CsMat;
use std::fs::File;
use std::time::Instant;
use std::{f64::consts::PI, io::Write};
use clap::Parser;

pub mod cli;
use crate::cli::{Cli, Command, MethodArg, BandParallelModeArg};

pub mod utils;
pub use utils::{BosonMode, Polarization, Vector3D};

pub mod matrix_elements;
pub use matrix_elements::LMMatrixElementArray;

pub mod visualization;
pub use visualization::{DielectricPlotType, plot_band_structure, plot_dielectric_function};

pub mod band_structure;
pub mod dielectric_function;

use crate::utils::PPConfig;

pub mod effective_modes;

pub mod logging;

pub mod visualization_dielectric;

pub mod hamiltonians;
use crate::hamiltonians::{PPHamiltonian, SingExBosons};

// Speed of light in atomic units
const C: f64 = 137.036;

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

    let n_rel = args.n_rel;
    let n_ex_eig = args.n_ex_eig;
    let n_eig = args.n_eig;

    let calculate = args.calculate;
    let zoomed = args.zoomed;

    let total_coupling = args.total_coupling;
    timed_println!("total_coupling = {}", total_coupling);

    let omega_0 = args.omega_0;
    let n_modes = args.n_modes;
    let coupling = total_coupling / (n_modes as f64);

    let qx_values = Array1::linspace(-args.q_max, args.q_max, n_modes);
    let qy_values = Array1::linspace(-args.q_max, args.q_max, n_modes);

    let qz = omega_0 / C;
    let n_fock = 2;

    let n_k_points = args.n_k_points;
    let n_bands = args.n_bands;

    let pn_coupling = args.pn_coupling;
    let pn_omega = args.pn_omega;
    let pn_k = arr1(&[0.0, args.pn_k_y]);
    let pn_n_fock = args.pn_n_fock;

    let q_values: Vec<Vector3D> = qx_values
        .iter()
        .flat_map(|qx| {
            qy_values.iter().map(|qy| Vector3D {
                x: *qx,
                y: *qy,
                z: qz,
            })
        })
        .collect();

    // Create uniform coupling vector (same coupling for all modes)
    let lm_couplings = vec![coupling; q_values.len()];

    let pp_config = PPConfig::new(
        e_config,
        n_rel,
        n_ex_eig,
        lm_couplings.clone(),
        omega_0,
        q_values,
        pn_coupling,
        pn_omega,
        pn_k,
        pn_n_fock,
        n_k_points,
        n_bands,
    );

    // Create photon modes
    let photon_modes: Vec<BosonMode> = qx_values
        .iter()
        .flat_map(|qx| {
            qy_values.iter().map(|qy| {
                BosonMode::new(
                    Vector3D {
                        x: *qx,
                        y: *qy,
                        z: qz,
                    },
                    Polarization::TE,
                    n_fock,
                )
                .clone()
            })
        })
        .collect();

    // Define high-symmetry points in k-space for band structure calculation
    let sym_points = vec![vec![0.0, 0.0], vec![PI / a0, 0.0], vec![PI / a0, PI / a0]];
    let sym_labels = vec!["Γ".to_string(), "X".to_string(), "M".to_string()];

    timed_println!("Running with {} modes", photon_modes.len());

    match method {
        Method::Sparse => {
            timed_println!("Using sparse methods for light-matter coupling.");

            // Start timing the band structure calculation
            let start_time = Instant::now();

            let mut pp: PPHamiltonian<CsMat<c64>, SingExBosons<CsMat<c64>>> =
                PPHamiltonian::from_config(&pp_config);

            let (k_points_color, eigvals_list_color, photonic_character_list);

            if calculate {
                // Build the PP Hamiltonian
                let time1 = Instant::now();

                // Calculate the band structure
                timed_println!("Calculating {} eigenstates", n_eig);
                timed_println!("Calculating color-coded band structure...");
                (k_points_color, eigvals_list_color, photonic_character_list) =
                    run_bandstructure_parallel_mode(
                        &mut pp,
                        args.parallel,
                        n_k_points,
                        n_eig,
                        &sym_points,
                        &pp_config,
                        args.use_cached,
                        zoomed,
                        chunk_index,
                        n_chunks,
                    )?;
                timed_println!(
                    "Calculated Band Structure in {:?}",
                    Instant::now().duration_since(time1)
                );
            } else {
                // Load bandstructure
                (k_points_color, eigvals_list_color, photonic_character_list) =
                    band_structure::load_color_band_structure(
                        &mut pp, n_k_points, n_eig, &pp_config, zoomed,
                    );
            }

            // End timing and print duration
            let end_time = Instant::now();
            let duration = end_time.duration_since(start_time);
            timed_println!("Calc dispersion in {:?}", duration);

            // Plot using gnuplot (only on rank 0 which has the data)
            let mut save_path = format!("sing_ex_color_disp_plot_{}_n{}", total_coupling, n_modes);
            save_path += if zoomed { "_zoomed.pdf" } else { ".pdf" };

            if !k_points_color.is_empty()
                && crate::utils::is_mpi_rank0_env()
                && !matches!(args.parallel, BandParallelModeArg::DistributedChunks)
            {
                if let Err(e) = band_structure::plot_color_band_structure(
                    &k_points_color,
                    &sym_points,
                    &sym_labels,
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

        Method::Dense => {
            timed_println!("Using dense methods for light-matter coupling.");

            // Start timing the band structure calculation
            let start_time = Instant::now();

            let mut pp: PPHamiltonian<Array2<c64>, SingExBosons<Array2<c64>>> =
                PPHamiltonian::from_config(&pp_config);

            let (k_points_color, eigvals_list_color, photonic_character_list);

            if calculate {
                // Build the PP Hamiltonian
                let time1 = Instant::now();

                // Calculate the band structure
                timed_println!("Calculating {} eigenstates", n_eig);
                timed_println!("Calculating color-coded band structure...");
                (k_points_color, eigvals_list_color, photonic_character_list) =
                    run_bandstructure_parallel_mode(
                        &mut pp,
                        args.parallel,
                        n_k_points,
                        n_eig,
                        &sym_points,
                        &pp_config,
                        args.use_cached,
                        zoomed,
                        chunk_index,
                        n_chunks,
                    )?;
                timed_println!(
                    "Calculated Band Structure in {:?}",
                    Instant::now().duration_since(time1)
                );
            } else {
                // Load bandstructure
                (k_points_color, eigvals_list_color, photonic_character_list) =
                    band_structure::load_color_band_structure(
                        &mut pp, n_k_points, n_eig, &pp_config, zoomed,
                    );
            }

            // End timing and print duration
            let end_time = Instant::now();
            let duration = end_time.duration_since(start_time);
            timed_println!("Calc dispersion in {:?}", duration);

            // Plot using gnuplot (only on rank 0 which has the data)
            let mut save_path = format!("sing_ex_color_disp_plot_{}_n{}", total_coupling, n_modes);
            save_path += if zoomed { "_zoomed.pdf" } else { ".pdf" };

            if !k_points_color.is_empty()
                && crate::utils::is_mpi_rank0_env()
                && !matches!(args.parallel, BandParallelModeArg::DistributedChunks)
            {
                if let Err(e) = band_structure::plot_color_band_structure(
                    &k_points_color,
                    &sym_points,
                    &sym_labels,
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

        Method::LinOp => {
            timed_println!("Using dense methods for light-matter coupling.");

            // Start timing the band structure calculation
            let start_time = Instant::now();

            let mut pp: PPHamiltonian<ComposedOperator<c64>, SingExBosons<ComposedOperator<c64>>> =
                PPHamiltonian::from_config(&pp_config);

            let (k_points_color, eigvals_list_color, photonic_character_list);

            if calculate {
                // Build the PP Hamiltonian
                let time1 = Instant::now();

                // Calculate the band structure
                timed_println!("Calculating {} eigenstates", n_eig);
                timed_println!("Calculating color-coded band structure...");
                (k_points_color, eigvals_list_color, photonic_character_list) =
                    run_bandstructure_parallel_mode(
                        &mut pp,
                        args.parallel,
                        n_k_points,
                        n_eig,
                        &sym_points,
                        &pp_config,
                        args.use_cached,
                        zoomed,
                        chunk_index,
                        n_chunks,
                    )?;
                timed_println!(
                    "Calculated Band Structure in {:?}",
                    Instant::now().duration_since(time1)
                );
            } else {
                // Load bandstructure
                (k_points_color, eigvals_list_color, photonic_character_list) =
                    band_structure::load_color_band_structure(
                        &mut pp, n_k_points, n_eig, &pp_config, zoomed,
                    );
            }

            // End timing and print duration
            let end_time = Instant::now();
            let duration = end_time.duration_since(start_time);
            timed_println!("Calc dispersion in {:?}", duration);

            // Plot using gnuplot (only on rank 0 which has the data)
            let mut save_path = format!("sing_ex_color_disp_plot_{}_n{}", total_coupling, n_modes);
            save_path += if zoomed { "_zoomed.pdf" } else { ".pdf" };

            if !k_points_color.is_empty()
                && crate::utils::is_mpi_rank0_env()
                && !matches!(args.parallel, BandParallelModeArg::DistributedChunks)
            {
                if let Err(e) = band_structure::plot_color_band_structure(
                    &k_points_color,
                    &sym_points,
                    &sym_labels,
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
    }
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
    B: hamiltonians::Boson<M>,
{
    match parallel {
        BandParallelModeArg::Serial => {
            // Serial is equivalent to a single distributed chunk.
            band_structure::calculate_color_band_structure_distributed_chunk(
                pp,
                n_k_points,
                n_eig,
                sym_points,
                pp_config,
                use_cached,
                zoomed,
                0,
                1,
            )
        }

        BandParallelModeArg::KpointMpiLegacy => {
            band_structure::calculate_color_band_structure_mpi(
                pp,
                n_k_points,
                n_eig,
                sym_points,
                pp_config,
                use_cached,
                zoomed,
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

        BandParallelModeArg::GatherChunks => {
            band_structure::gather_color_band_structure_chunks(
                n_k_points,
                n_eig,
                pp_config,
                zoomed,
                n_chunks,
            )
        }
    }
}

/// Helper function to generate stacked area plots for photonic and phonon contributions
fn generate_stacked_plots(
    coupling_values: &Array1<f64>,
    all_mode_couplings: &Vec<Vec<f64>>,
    all_q_values: &Vec<Vec<Vector3D>>,
    reflectivities: &[f64],
    e_config: &ExcitonConfig,
    n_rel: usize,
    n_ex_eig: usize,
    omega_0: f64,
    d_omega: f64,
    omega_max: f64,
    pn_coupling: f64,
    pn_omega: f64,
    pn_k: &Array1<f64>,
    pn_n_fock: usize,
    n_eig: usize,
    omegas: &Array1<f64>,
    gamma: f64,
    save_dir: &str,
    zoomed: bool,
    omega_range: Option<(f64, f64)>,
) {
    if !crate::utils::is_mpi_rank0_env() {
        return;
    }

    timed_println!("Generating stacked photonic character plots...");

    // Load zero-coupling dielectric function for overlay (use first q_values set)
    let zero_coupling_imag = {
        let zero_couplings = vec![0.0; all_q_values[0].len()];
        let pp_config_zero = PPConfig::new_with_effective_modes(
            e_config.clone(),
            n_rel,
            n_ex_eig,
            zero_couplings,
            omega_0,
            all_q_values[0].clone(),
            pn_coupling,
            pn_omega,
            pn_k.clone(),
            pn_n_fock,
            1,
            1,
            d_omega,
            omega_max,
            reflectivities[0],
        );

        let pp_zero: PPHamiltonian<ComposedOperator<c64>, SingExBosons<ComposedOperator<c64>>> =
            PPHamiltonian::from_config(&pp_config_zero);

        let cache_filename_zero = dielectric_function::get_dielectric_cache_filename(
            &pp_zero,
            &pp_config_zero,
            save_dir,
            n_eig,
        );

        match dielectric_function::load_dielectric_cache(&cache_filename_zero) {
            Ok((pp_vals_zero, x_pp_zero, _, _, gs_idx_zero)) => {
                // Compute Im[ε] for zero coupling
                let ground_state_energy = pp_vals_zero[gs_idx_zero];
                let mut im_eps_zero = Array1::<f64>::zeros(omegas.len());

                for (w_idx, &omega) in omegas.iter().enumerate() {
                    let mut eps_contrib = c64::new(0.0, 0.0);

                    for i in 0..pp_vals_zero.len() {
                        if i == gs_idx_zero {
                            continue;
                        }
                        let delta_e = pp_vals_zero[i] - ground_state_energy;
                        let contrib =
                            x_pp_zero[i] * x_pp_zero[i] / (omega - delta_e + c64::i() * gamma);
                        eps_contrib += contrib;
                    }

                    im_eps_zero[w_idx] = (1.0 - 4.0 * PI * eps_contrib).im.abs();
                }

                Some(im_eps_zero)
            }
            Err(e) => {
                timed_println!(
                    "Warning: Could not load zero-coupling cache for overlay: {}",
                    e
                );
                None
            }
        }
    };

    for (idx, &total_coupling) in coupling_values.iter().enumerate() {
        if total_coupling.abs() < 1e-12 {
            continue; // Skip zero coupling
        }

        // Use the actual mode couplings that were used during computation
        let mode_couplings = &all_mode_couplings[idx];
        let q_values = &all_q_values[idx];
        let reflectivity = reflectivities[idx];

        // Create PP config for this coupling with effective mode parameters
        let pp_config_plot = PPConfig::new_with_effective_modes(
            e_config.clone(),
            n_rel,
            n_ex_eig,
            mode_couplings.clone(),
            omega_0,
            q_values.clone(),
            pn_coupling,
            pn_omega,
            pn_k.clone(),
            pn_n_fock,
            1,
            1,
            d_omega,
            omega_max,
            reflectivity,
        );

        let pp_plot: PPHamiltonian<ComposedOperator<c64>, SingExBosons<ComposedOperator<c64>>> =
            PPHamiltonian::from_config(&pp_config_plot);

        // Load cached eigenstate data
        let cache_filename = dielectric_function::get_dielectric_cache_filename(
            &pp_plot,
            &pp_config_plot,
            save_dir,
            n_eig,
        );

        match dielectric_function::load_dielectric_cache(&cache_filename) {
            Ok((pp_vals, x_pp, photonic_chars, phonon_numbers, gs_idx)) => {
                let n_photon_modes = q_values.len();

                // Plot photonic character stacked area
                let photonic_plot_name = format!(
                    "{}/dielectric_photonic_stacked_a0{}_V0{}_nrel{}_lmcoup{:.6}_nmodes{}_pncoup{}_npn{}_neig{}_gamma{:.1e}.svg",
                    save_dir,
                    e_config.a0,
                    e_config.v0,
                    n_rel,
                    total_coupling,
                    n_photon_modes,
                    pn_coupling,
                    pn_n_fock,
                    n_eig,
                    gamma
                );

                match visualization_dielectric::plot_dielectric_stacked_photonic(
                    omegas,
                    &pp_vals,
                    &x_pp,
                    &photonic_chars,
                    gs_idx,
                    gamma,
                    &photonic_plot_name,
                    zoomed,
                    omega_range,
                    Some(q_values),
                    zero_coupling_imag.as_ref(),
                ) {
                    Ok(_) => timed_println!(
                        "Successfully plotted photonic stacked area for coupling {:.6}",
                        total_coupling
                    ),
                    Err(e) => timed_println!("Error plotting photonic stacked area: {}", e),
                }

                // Plot phonon Fock state stacked area
                if pn_n_fock > 0 {
                    let phonon_plot_name = format!(
                        "{}/dielectric_phonon_stacked_a0{}_V0{}_nrel{}_lmcoup{:.6}_nmodes{}_pncoup{}_npn{}_neig{}_gamma{:.1e}.svg",
                        save_dir,
                        e_config.a0,
                        e_config.v0,
                        n_rel,
                        total_coupling,
                        n_photon_modes,
                        pn_coupling,
                        pn_n_fock,
                        n_eig,
                        gamma
                    );

                    match visualization_dielectric::plot_dielectric_stacked_phonon(
                        omegas,
                        &pp_vals,
                        &x_pp,
                        &phonon_numbers,
                        gs_idx,
                        pn_n_fock,
                        gamma,
                        &phonon_plot_name,
                        zoomed,
                        omega_range,
                        Some(q_values),
                        zero_coupling_imag.as_ref(),
                    ) {
                        Ok(_) => timed_println!(
                            "Successfully plotted phonon stacked area for coupling {:.6}",
                            total_coupling
                        ),
                        Err(e) => timed_println!("Error plotting phonon stacked area: {}", e),
                    }
                }
            }
            Err(e) => timed_println!(
                "Could not load cached data for coupling {:.6}: {}",
                total_coupling,
                e
            ),
        }
    }
}

/// Main function demonstrating the use of the light-matter coupling framework
/// Calculate dielectric function for a range of coupling strengths
fn run_dielectric_scan(e_config: ExcitonConfig, args: cli::DielectricArgs) -> Result<()> {
    timed_println!("Starting dielectric function scan...");

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
    timed_println_rank0!("  omega scan      = [{}, {}], n={}", args.omega_min_scan, args.omega_max_scan, args.n_omega_points);
    timed_println_rank0!("  output_dir      = {}", save_dir);

    // Generate effective modes and couplings for each reflectivity.
    let mut all_q_values = Vec::new();
    let mut all_mode_couplings = Vec::new();
    let mut total_coupling_values = Vec::new();

    for &reflectivity in reflectivities.iter() {
        let (q_values, mode_couplings) =
            effective_modes::generate_modes(d_omega, n_modes, omega_0, omega_max, reflectivity);

        let total_coupling: f64 = mode_couplings.iter().sum();

        timed_println_rank0!(
            "Reflectivity {}: generated {} modes with total coupling {:.6}",
            reflectivity,
            q_values.len(),
            total_coupling
        );

        all_q_values.push(q_values);
        all_mode_couplings.push(mode_couplings);
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
        save_dir,
        e_config.a0,
        e_config.v0,
        n_rel,
        n_modes,
        pn_coupling,
        pn_n_fock,
        suffix,
    );

    let imag_filename = format!(
        "{}/dielectric_imag_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}{}",
        save_dir,
        e_config.a0,
        e_config.v0,
        n_rel,
        n_modes,
        pn_coupling,
        pn_n_fock,
        suffix,
    );

    let both_filename = format!(
        "{}/dielectric_both_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}{}",
        save_dir,
        e_config.a0,
        e_config.v0,
        n_rel,
        n_modes,
        pn_coupling,
        pn_n_fock,
        suffix,
    );

    if !calculate {
        if !crate::utils::is_mpi_rank0_env() {
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
        .or_else(|| crate::utils::env_usize(&["REFLECTIVITY_INDEX", "SLURM_ARRAY_TASK_ID"]))
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
    let mode_couplings = &all_mode_couplings[refl_idx];
    let reflectivity = reflectivities[refl_idx];

    timed_println_rank0!(
        "Selected reflectivity index {}: R={}, modes={}, total coupling={:.6}",
        refl_idx,
        reflectivity,
        q_values.len(),
        total_coupling
    );

    let start_time = Instant::now();

    let pp_config = PPConfig::new_with_effective_modes(
        e_config.clone(),
        n_rel,
        n_ex_eig,
        mode_couplings.clone(),
        omega_0,
        q_values.clone(),
        pn_coupling,
        pn_omega,
        pn_k.clone(),
        pn_n_fock,
        1,
        1,
        d_omega,
        omega_max,
        reflectivity,
    );

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
        &pp,
        &pp_config,
        save_dir,
        n_eig,
        gamma,
    );

    if crate::utils::is_mpi_rank0_env() {
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

fn run_matrix_elem_calc_cli(args: cli::MatrixElemCalcArgs) -> Result<()> {
    // This command is intended as a shared-cache CPU precompute.
    // Avoid many MPI ranks writing the same cache files.
    if !crate::utils::is_mpi_rank0_env() {
        return Ok(());
    }

    let mut e_config = args.exciton.to_config();

    // Make the exciton save_dir consistent with the CLI.
    e_config.save_dir = args.exciton.save_dir.clone();

    let refl_idx = args
        .reflectivity_index
        .or_else(|| crate::utils::env_usize(&["REFLECTIVITY_INDEX", "SLURM_ARRAY_TASK_ID"]))
        .unwrap_or(0);

    if refl_idx >= args.reflectivities.len() {
        panic!(
            "reflectivity_index {} out of range 0..{}",
            refl_idx,
            args.reflectivities.len()
        );
    }

    let reflectivity = args.reflectivities[refl_idx];

    timed_println!("Starting standalone LM matrix-element precompute");
    timed_println!("  n_rel             = {}", args.n_rel);
    timed_println!("  n_ex_eig          = {}", args.n_ex_eig);
    timed_println!("  omega_0           = {}", args.omega_0);
    timed_println!("  omega_max         = {}", args.omega_max);
    timed_println!("  d_omega           = {}", args.d_omega);
    timed_println!("  n_modes side      = {}", args.n_modes);
    timed_println!("  target modes      = {}", args.n_modes * args.n_modes);
    timed_println!("  reflectivity idx  = {}", refl_idx);
    timed_println!("  reflectivity      = {}", reflectivity);
    timed_println!("  cache_dir         = {}", args.cache_dir);
    timed_println!("  force_recompute   = {}", args.force_recompute);

    std::fs::create_dir_all(&args.cache_dir).unwrap_or_else(|e| {
        panic!("Failed to create cache directory {}: {}", args.cache_dir, e);
    });

    let (q_values, mode_couplings) = effective_modes::generate_modes(
        args.d_omega,
        args.n_modes,
        args.omega_0,
        args.omega_max,
        reflectivity,
    );

    let total_coupling: f64 = mode_couplings.iter().sum();

    timed_println!(
        "Generated {} effective photon modes with total coupling {:.8}",
        q_values.len(),
        total_coupling
    );

    let pn_k = arr1(&[args.pn_k_x, 0.0]);

    let mut pp_config = PPConfig::new_with_effective_modes(
        e_config,
        args.n_rel,
        args.n_ex_eig,
        mode_couplings,
        args.omega_0,
        q_values,
        args.pn_coupling,
        args.pn_omega,
        pn_k,
        args.pn_n_fock,
        1,
        1,
        args.d_omega,
        args.omega_max,
        reflectivity,
    );

    pp_config.cache_dir = Some(args.cache_dir.clone());

    timed_println!(
        "PPConfig hash for matrix-element cache: {:016x}",
        pp_config.get_config_hash()
    );

    // Use the sparse backend for the precompute command.
    // This is a CPU/Rayon precompute and should usually be run as one process
    // with many Rayon threads, not as an MPI/SLEPc job.
    let mut pp: PPHamiltonian<CsMat<c64>, SingExBosons<CsMat<c64>>> =
        PPHamiltonian::from_config(&pp_config);

    pp.polariton
        .compute_lm_matrix_elements_only(None, args.force_recompute)?;

    timed_println!("Standalone LM matrix-element precompute finished");

    Ok(())
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
        .or_else(|| crate::utils::env_usize(&["K_CHUNK_INDEX", "SLURM_ARRAY_TASK_ID"]))
        .unwrap_or(0);

    let n_chunks = args
        .k_n_chunks
        .or_else(|| crate::utils::env_usize(&["K_N_CHUNKS", "SLURM_ARRAY_TASK_COUNT"]))
        .unwrap_or(1);

    run_bandstructure_with_args(method, e_config, args, chunk_index, n_chunks)
}

fn run_dielectric_cli(args: cli::DielectricArgs) -> Result<()> {
    let e_config = args.exciton.to_config();
    run_dielectric_scan(e_config, args)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

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