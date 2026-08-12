use exciton_2d::ExcitonConfig;
use matrix_ops::{ComposedOperator, Result};
use ndarray::{Array1, Array2, arr1};
use ndarray_linalg::c64;
use sprs::CsMat;
use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;
use std::{f64::consts::PI, io::Write};

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

fn run_bandstructure(method: Method, e_config: ExcitonConfig) -> Result<()> {
    // Create the 2D exciton system
    let a0 = (&e_config).a0;
    let v0 = (&e_config).v0;

    // Parameters for diagonalization
    let n_rel = 16; // Number of relative coordinate states
    let n_ex_eig = 32; // Number of exciton eigenstates
    let n_eig = 512; // Number PP eigenstates

    let calculate = true;

    let zoomed = true;

    // Light-matter coupling parameters
    let total_coupling = 0.003; // Coupling strength
    timed_println!("total_coupling = {}", total_coupling);
    let omega_0 = 0.15; // Photon frequency
    let n_modes = 23;
    let coupling = total_coupling / (n_modes as f64);

    // Define photon modes
    // let qx_values = vec![0.0]; // Photon wavevectors along x
    let qx_values = Array1::linspace(-0.001, 0.001, n_modes); // Photon wavevectors along x
    let qy_values = Array1::linspace(-0.001, 0.001, n_modes);
    let qz = omega_0 / C;
    let n_fock = 2; // Dimension of Fock space

    let n_k_points = 128; // Number of k-points along the path
    let n_bands = 16;

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

    // Define phonon parameters
    let pn_coupling = 0.0;
    let pn_omega = 0.0075; // 1500 cm^-1
    let pn_k = arr1(&[0., 0.05]); // 1 nm^1
    let pn_n_fock = 0;

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
                    band_structure::calculate_color_band_structure_mpi(
                        &mut pp,
                        n_k_points,
                        n_eig,
                        &sym_points,
                        &pp_config,
                        false,
                        zoomed,
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

            if !k_points_color.is_empty() {
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
                    band_structure::calculate_color_band_structure_mpi(
                        &mut pp,
                        n_k_points,
                        n_eig,
                        &sym_points,
                        &pp_config,
                        false,
                        zoomed,
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

            if !k_points_color.is_empty() {
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
                    band_structure::calculate_color_band_structure_mpi(
                        &mut pp,
                        n_k_points,
                        n_eig,
                        &sym_points,
                        &pp_config,
                        false,
                        zoomed,
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

            if !k_points_color.is_empty() {
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
fn run_dielectric_scan(e_config: ExcitonConfig) -> Result<()> {
    timed_println!("Starting dielectric function scan...");

    // Check if we should calculate or just plot
    let calculate = true; // Set to false to skip calculation and just plot
    let use_cached = false; // Set to true to use cached diagonalization results (allows testing different gamma values)

    let zoomed = false;

    // Parameters for diagonalization
    let n_rel = 16; // Number of relative coordinate states
    let n_ex_eig = 16; // Number of exciton eigenstates

    let omega_0 = 0.18; // Photon frequency
    let n_modes = 32;

    // Define photon modes based on effective modes paper
    let omega_max = 0.2;
    let reflectivities = [0.0, 0.999999, 0.9999999];
    let d_omega = 1e-4;

    // Generate effective modes and their couplings for each reflectivity
    let mut all_q_values = Vec::new();
    let mut all_mode_couplings = Vec::new();
    let mut total_coupling_values = Vec::new();
    for &reflectivity in reflectivities.iter() {
        let (q_values, mode_couplings) =
            effective_modes::generate_modes(d_omega, n_modes, omega_0, omega_max, reflectivity);

        let total_coupling: f64 = mode_couplings.iter().sum();
        timed_println!(
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

    // Validate that all reflectivities generate the same number of modes
    let n_modes_first = all_q_values[0].len();
    for (idx, q_vals) in all_q_values.iter().enumerate() {
        if q_vals.len() != n_modes_first {
            timed_println!(
                "Warning: Reflectivity {} generated {} modes, but first generated {}. This may cause issues.",
                reflectivities[idx],
                q_vals.len(),
                n_modes_first
            );
        }
    }

    // Define phonon parameters
    // let pn_coupling = 0.01; // 114 cm^-1
    // let pn_coupling = 0.004385964912; // 50 cm^-1
    // let pn_coupling = 0.008771929825; // 100 cm^-1
    let pn_coupling = 0.0;
    let pn_omega = 0.0025; // 500 cm^-1
    // let pn_coupling = 0.001; // 11.4 cm^-1
    // let pn_coupling = 0.0005; // 11.4 cm^-1
    let pn_k = arr1(&[0., 0.05]); // 1 nm^1
    let pn_n_fock = 1;

    // Parameters for dielectric function calculation
    let n_eig = 1024; // Number of eigenstates to use
    let omegas = Array1::linspace(0.0, 0.3, 100000); // Frequency range
    // let gamma = 5e-5;  // Broadening parameter ~0.9 meV
    let gamma = d_omega;

    let omega_range;
    if !zoomed {
        omega_range = Some((0.182, 0.202));
    } else {
        // omega_range = Some((0.174,0.18));
        omega_range = Some((0.183, 0.188));
    }

    // Create directory for saving results
    let save_dir = "dielectric_scan/effective_modes/";
    std::fs::create_dir_all(save_dir).unwrap_or_else(|_| {
        timed_println!(
            "Directory {} already exists or couldn't be created",
            save_dir
        );
    });

    // Create more descriptive filenames
    let mut real_filename = format!(
        "dielectric_real_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}",
        e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock
    );
    let mut imag_filename = format!(
        "dielectric_imag_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}",
        e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock
    );
    let mut both_filename = format!(
        "dielectric_both_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}",
        e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock
    );
    if !zoomed {
        real_filename += ".svg";
        imag_filename += ".svg";
        both_filename += ".svg";
    } else {
        real_filename += "_zoomed.svg";
        imag_filename += "_zoomed.svg";
        both_filename += "_zoomed.svg";
    }

    if !calculate {
        if !crate::utils::is_mpi_rank0_env() {
            return Ok(());
        }

        // Just plot the existing data
        timed_println!("Plotting dielectric function from existing data...");
        let q_values = &all_q_values[0];

        // Create reference parameters to match against
        let ref_params = visualization::DielectricParams {
            a0: e_config.a0,
            v0: e_config.v0,
            n_rel,
            lm_coupling: 0.0, // This will be matched separately using total_coupling_values
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
            Some(&q_values),
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
            Some(&q_values),
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
        "Starting dielectric function scan for {} total coupling values",
        total_coupling_values.len()
    );

    // Select exactly one reflectivity case for this process group.
    //
    // Important:
    // PETSc/SLEPc eigensolvers are collective over PETSC_COMM_WORLD. Therefore,
    // all MPI ranks in this job must build and diagonalize the same Hamiltonian.
    // Do not distribute reflectivities over MPI ranks inside one PETSc
    // communicator. Use a Slurm job array or separate job submission instead.
    let refl_idx: usize = env::var("REFLECTIVITY_INDEX")
        .or_else(|_| env::var("SLURM_ARRAY_TASK_ID"))
        .unwrap_or_else(|_| "0".to_string())
        .parse()
        .expect("REFLECTIVITY_INDEX/SLURM_ARRAY_TASK_ID must parse as usize");

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

    timed_println!("Using ComposedOperator + distributed SLEPc shell backend");

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

    timed_println!(
        "Completed reflectivity index {} in {:?}",
        refl_idx,
        start_time.elapsed()
    );

    Ok(())
}

fn main() -> Result<()> {
    // Get the number of available cores
    let num_threads = std::env::var("OMP_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    // let num_threads = 16;

    // Configure Rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    // Create exciton configuration
    let e_config = ExcitonConfig {
        n_kappa: 7,                               // Number of bands
        a0: 16.0,                                 // Lattice constant
        v0: 0.05,                                 // Potential strength
        k: vec![0.0, 0.0],                        // Initial k-vector
        m_e: 0.2,                                 // Electron effective mass
        m_h: 0.2,                                 // Hole effective mass
        verbose: false,                           // Verbose output flag
        grid_size: 64,                            // Grid size for real-space calculations
        box_size: 3.0,                            // Real-space box size
        save_dir: PathBuf::from("saved_data_lm"), // Directory to save data
        recalculate: false,                       // Whether to recalculate existing data
        n_jobs: 32,                               // Number of parallel jobs
        use_gpu: false,                           // Whether to use GPU acceleration
        batch_size: 32,                           // Batch size for calculations
    };
    // davidson::davidson_test();

    // Choose which function to run
    let run_mode = "dielectric_scan"; // Options: "bandstructure", "dielectric_scan"

    match run_mode {
        "bandstructure" => {
            // run_bandstructure(Method::LinOp, e_config.clone());
            // run_bandstructure(Method::SingExLegacy, e_config.clone());
            run_bandstructure(Method::Sparse, e_config.clone())?;
            // run_bandstructure(Method::Dense, e_config.clone());
        }
        "dielectric_scan" => {
            run_dielectric_scan(e_config.clone())?;
        }
        _ => {
            timed_println!("Unknown run mode: {}", run_mode);
        }
    }

    Ok(())
}
