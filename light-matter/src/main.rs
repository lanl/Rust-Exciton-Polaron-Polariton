use exciton_2d::{Exciton2DGridSystem, ExcitonConfig};
use sprs::CsMat;
use std::{f64::consts::PI, io::Write};
use std::time::Instant;
use ndarray::{arr1, Array1, Array2, s};
use rust_slepc_bindings::{initialize, finalize};
use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use ndarray_linalg::c64;

pub mod sparse_methods;

pub mod utils;
pub use utils::{Vector3D, BosonMode, Polarization};

pub mod matrix_elements;
pub use matrix_elements::LMMatrixElementArray;

pub mod visualization;
pub use visualization::{plot_band_structure, plot_dielectric_function, DielectricPlotType};

pub mod color_band_plot;
pub mod dielectric_function;
pub mod band_structure;

use crate::utils::{ComposedOperator, PPConfig};

pub mod sing_ex;
pub mod lin_op;

pub mod dense_methods;

pub mod slepc;

pub mod slepc_instrumented;

pub mod logging;

pub mod visualization_dielectric;

pub mod hamiltonians;
use crate::hamiltonians::{BosonicHamiltonian, ExcitonicHamiltonian, FullBosons, LMHamiltonian, PPHamiltonian, SingExBosons};

// Speed of light in atomic units
const C: f64 = 137.036;

pub enum Method {
    Sparse,
    Dense,
    LinOp,
    SingExLegacy,
}

fn run_bandstructure(method: Method, e_config: ExcitonConfig) {

    // Create the 2D exciton system
    let a0 = (&e_config).a0;
    let v0 = (&e_config).v0;
    let exciton = Exciton2DGridSystem::from_config(e_config.clone());
    
    // Parameters for diagonalization
    let n_rel = 16;      // Number of relative coordinate states
    let n_ex_eig = 32;  // Number of exciton eigenstates
    let n_eig = 512; // Number PP eigenstates

    let calculate = true;
        
    let zoomed = true;
    
    // Light-matter coupling parameters
    let total_coupling = 0.003 ;  // Coupling strength
    timed_println!("total_coupling = {}", total_coupling);
    let omega_0 = 0.15;    // Photon frequency
    let n_modes = 23;
    let coupling = total_coupling / (n_modes as f64);

    // Define photon modes
    // let qx_values = vec![0.0]; // Photon wavevectors along x
    let qx_values = Array1::linspace(-0.001, 0.001, n_modes); // Photon wavevectors along x
    let qy_values = Array1::linspace(-0.001, 0.001, n_modes);
    let qz = omega_0 / C;
    let n_fock = 2;      // Dimension of Fock space

    let n_k_points = 1024;  // Number of k-points along the path
    let n_bands = 16;

    let q_values: Vec<Vector3D> = qx_values.iter().flat_map(|qx| {
        qy_values.iter().map(|qy| {
            Vector3D { x: *qx, y: *qy, z: qz }
        })}).collect();
    

    // Define phonon parameters
    let pn_coupling = 0.0;
    let pn_omega = 0.0075; // 1500 cm^-1
    let pn_k = arr1(&[0.,0.05]); // 1 nm^1
    let pn_n_fock = 0;

    let pp_config = PPConfig{
        e_config, n_rel, n_ex_eig, lm_coupling: coupling, omega_0, q_values,
        pn_coupling, pn_omega, pn_k, pn_n_fock, n_k_points, n_bands
    };

    // Create photon modes
    let photon_modes: Vec<BosonMode> = qx_values.iter().flat_map(|qx| {
            qy_values.iter().map( |qy| {
                BosonMode::new(Vector3D { x: *qx, y: *qy, z: qz }, Polarization::TE, n_fock).clone()
            })
        }).collect();

    // Define high-symmetry points in k-space for band structure calculation
    let sym_points = vec![ vec![0.0,0.0], vec![PI / a0, 0.0], vec![PI / a0, PI / a0]];
    let sym_labels = vec!["Γ".to_string(), "X".to_string(), "M".to_string()];
    
    timed_println!("Running with {} modes", photon_modes.len());

    match method {
        Method::Sparse => {

            timed_println!("Using sparse methods for light-matter coupling.");

            // // Create light-matter system
            // let photon = BosonicHamiltonian::<CsMat<c64>, SingExBosons<CsMat<c64>>>::new(photon_modes);
            // let exciton = ExcitonicHamiltonian::new(exciton.full_ham, n_ex_eig, n_rel);
            // let light_matter = LMHamiltonian::new(exciton, photon, coupling);

            // Start timing the band structure calculation
            let start_time = Instant::now();
            
            // // Create band structure calculator
            // let mut band_structure = sparse_methods::BandStructureCalculatorLM::new(
            //     light_matter, a0, v0, "lm_data".to_string(), sym_points.clone());

            // // Using sparse matrices and SLEPc to calculate eigenvalues
            // let (k_points, eigvals_list) = band_structure.calculate_lm_band_structure_sparse(n_k_points, n_eig);
            
            let mut pp: PPHamiltonian<CsMat<c64>, SingExBosons<CsMat<c64>>> = PPHamiltonian::from_config(&pp_config);

            let (k_points_color, eigvals_list_color, photonic_character_list);

            if calculate {
                // Build the PP Hamiltonian
                // pp.initialize();
                let time1 = Instant::now();
                // let gen_time = time1.duration_since(start_time);
                // timed_println!("Built PPHamitonian in {:?}", gen_time);

                // Calculate the band structure
                timed_println!("Calculating {} eigenstates", n_eig);
                timed_println!("Calculating color-coded band structure...");
                (k_points_color, eigvals_list_color, photonic_character_list) =
                    band_structure::calculate_color_band_structure_mpi_simple(&mut pp, n_k_points, n_eig, &sym_points, &pp_config, false, zoomed);
                timed_println!("Calculated Band Structure in {:?}", Instant::now().duration_since(time1));
            }
            else {
                // Load bandstructure
                (k_points_color, eigvals_list_color, photonic_character_list) = 
                    band_structure::load_color_band_structure(
                    &mut pp, n_k_points, n_eig, &pp_config, zoomed);
            }


            // End timing and print duration
            let end_time = Instant::now();
            let duration = end_time.duration_since(start_time);
            timed_println!("Calc dispersion in {:?}", duration);

             // Plot using gnuplot (only on rank 0 which has the data)
            let mut save_path = format!("sing_ex_color_disp_plot_{}_n{}",total_coupling, n_modes);
            save_path += if zoomed {"_zoomed.pdf"} else {".pdf"};

            if !k_points_color.is_empty() {
                if let Err(e) = band_structure::plot_color_band_structure(
                    &k_points_color, &sym_points, &sym_labels, &eigvals_list_color,
                    &photonic_character_list, n_eig, a0, &save_path, v0, coupling, zoomed) {
                    timed_println!("Error plotting color-coded band structure: {}", e);
                }
            }

        },

        Method::Dense => {
            
            timed_println!("Using dense methods for light-matter coupling.");

            // Create light-matter system
            let photon = BosonicHamiltonian::<Array2<c64>, FullBosons<Array2<c64>>>::new(photon_modes);
            let exciton = ExcitonicHamiltonian::new(exciton.full_ham, n_ex_eig, n_rel);
            let light_matter = LMHamiltonian::new(exciton, photon, coupling);

            // Start timing the band structure calculation
            let start_time = Instant::now();
            
            // Create band structure calculator
            let mut band_structure = dense_methods::BandStructureCalculatorLM::new(
                light_matter, a0, v0, "lm_data".to_string(), sym_points.clone());
            
            // Using sparse matrices and ARPACK to calculate 40 lowest eigenvalues
            let (k_points, eigvals_list) = band_structure.calculate_lm_band_structure(n_k_points);
            
            // End timing and print duration
            let end_time = Instant::now();
            let duration = end_time.duration_since(start_time);
            timed_println!("Calc dispersion in {:?}", duration);

            // Plot the band structure
            plot_band_structure(
                &k_points, sym_points, sym_labels, &eigvals_list, n_bands, a0, 
                Some("disp_plot.png"), false, v0, coupling);
        },

        Method::LinOp => {
            // let n_eig = 128;
            let start_time = Instant::now();

            // // Diagonalization test:
            // timed_println!("Creating PPHamiltonian from config...");
            // let time0 = Instant::now();
            // let mut pp: PPHamiltonian<ComposedOperator<c64>, SingExBosons<ComposedOperator<c64>>> = PPHamiltonian::from_config(&pp_config);
            // timed_println!("Created PPHamiltonian in {:?}", time0.elapsed());
            
            // timed_println!("Initializing PPHamiltonian...");
            // let time_init = Instant::now();
            // pp.initialize();
            // timed_println!("Initialized PPHamiltonian in {:?}", time_init.elapsed());

            // timed_println!("Building PP Hamiltonian matrix...");
            // let time_build = Instant::now();
            // let ham = pp.build_pp_hamiltonian(None);
            // timed_println!("Built PP Hamiltonian matrix in {:?}", time_build.elapsed());

            // let time1 = Instant::now();
            // let gen_time = time1.duration_since(start_time);
            // timed_println!("Total time to build PPHamitonian: {:?}", gen_time);

            // timed_println!("Starting diagonalization with {} eigenvalues...", n_eig);
            // let time_diag = Instant::now();
            // let (e_vals, _) = ham.diagonalize(Some(n_eig), None, None);

            // let diag_time = time_diag.elapsed();
            // timed_println!("Diagonalized Hamiltonian in {:?}", diag_time);

            // let total_time = Instant::now().duration_since(start_time);
            // timed_println!("Total time (including diagonalization): {:?}", total_time);

            // timed_println!("First ten energies:");
            // println!("{:?}", e_vals.slice(s![0..10]));

            // NEW IMPLEMENTATION:
            
            let mut pp: PPHamiltonian<ComposedOperator<c64>, SingExBosons<ComposedOperator<c64>>> = PPHamiltonian::from_config(&pp_config);

            let (k_points_color, eigvals_list_color, photonic_character_list);

            if calculate {
                // Build the PP Hamiltonian
                // pp.initialize(None);
                let time1 = Instant::now();
                // let gen_time = time1.duration_since(start_time);
                // timed_println!("Built PPHamitonian in {:?}", gen_time);

                // Calculate the band structure
                timed_println!("Calculating {} eigenstates", n_eig);
                timed_println!("Calculating color-coded band structure...");
                (k_points_color, eigvals_list_color, photonic_character_list) =
                    band_structure::calculate_color_band_structure_mpi_simple(&mut pp, n_k_points, n_eig, &sym_points, &pp_config, false, zoomed);
                timed_println!("Calculated Band Structure in {:?}", Instant::now().duration_since(time1));
            }
            else {
                // Load bandstructure
                (k_points_color, eigvals_list_color, photonic_character_list) = 
                    band_structure::load_color_band_structure(
                    &mut pp, n_k_points, n_eig, &pp_config, zoomed);
            }


            // End timing and print duration
            let end_time = Instant::now();
            let duration = end_time.duration_since(start_time);
            timed_println!("Calc dispersion in {:?}", duration);

             // Plot using gnuplot (only on rank 0 which has the data)
            let mut save_path = format!("sing_ex_color_disp_plot_{}_n{}",total_coupling, n_modes);
            save_path += if zoomed {"_zoomed.pdf"} else {".pdf"};

            if !k_points_color.is_empty() {
                if let Err(e) = band_structure::plot_color_band_structure(
                    &k_points_color, &sym_points, &sym_labels, &eigvals_list_color,
                    &photonic_character_list, n_eig, a0, &save_path, v0, coupling, zoomed) {
                    timed_println!("Error plotting color-coded band structure: {}", e);
                }
            }

        }

        Method::SingExLegacy => {
            timed_println!("Using Mat-Vec methods for light-matter coupling in single photon limit.");
            
            let start_time = Instant::now();

            let (k_points_color, eigvals_list_color, photonic_character_list);

            
            let mut pp_wrapper = sing_ex::SingExWrapper::new(pp_config.clone());
            pp_wrapper.generate_pp();
            let mut pp = pp_wrapper.polaron_polariton.unwrap();

            if calculate {
                timed_println!("Calculating {} eigenstates", n_eig);

                let time1 = Instant::now();
                let gen_time = time1.duration_since(start_time);
                timed_println!("Precomputed matrix elements in {:?}", gen_time);
                            
                // Calculate and plot color-coded band structure
                timed_println!("Calculating color-coded band structure...");
                (k_points_color, eigvals_list_color, photonic_character_list) = 
                    color_band_plot::calculate_color_band_structure_mpi_simple(
                        &mut pp, n_k_points, n_eig, &sym_points, &pp_config, true, zoomed);

                timed_println!("Calculated Band Structure in {:?}", Instant::now().duration_since(time1));
            }
            else {
                (k_points_color, eigvals_list_color, photonic_character_list) = 
                    color_band_plot::load_color_band_structure(
                    &mut pp, n_k_points, n_eig, &pp_config, zoomed);
            }
                
            // Plot using gnuplot (only on rank 0 which has the data)
            let mut save_path = format!("sing_ex_color_disp_plot_{}_n{}",total_coupling, n_modes);
            save_path += if zoomed {"_zoomed.pdf"} else {".pdf"};

            if !k_points_color.is_empty() {
                if let Err(e) = color_band_plot::plot_color_band_structure(
                    &k_points_color, &sym_points, &sym_labels, &eigvals_list_color,
                    &photonic_character_list, n_eig, a0, &save_path, v0, coupling, zoomed) {
                    timed_println!("Error plotting color-coded band structure: {}", e);
                }
            }
        } 
    }
}

/// Main function demonstrating the use of the light-matter coupling framework
/// Calculate dielectric function for a range of coupling strengths
fn run_dielectric_scan(e_config: ExcitonConfig) {
    // Check if we should calculate or just plot
    let calculate = true; // Set to false to skip calculation and just plot
    let use_cached = true; // Set to true to use cached diagonalization results (allows testing different gamma values)

    let zoomed = false;

    // Parameters for diagonalization
    let n_rel = 16;      // Number of relative coordinate states
    let n_ex_eig = 16;  // Number of exciton eigenstates
    
    // Light-matter coupling parameters
    // let coupling_values = Array1::linspace(0.00, 0.05, 1); // Scan from weak to strong coupling
    let total_coupling_values = Array1::from_vec(vec![0.0, 0.0035, 0.006]); // Scan from weak to strong coupling
    // let total_coupling_values = Array1::from_vec(vec![0.0043, 0.0044, 0.0045, 0.0046, 0.0047, 0.0048]);

    let omega_0 = 0.175;    // Photon frequency
    let n_modes = 45;
    let coupling_values = total_coupling_values /( (n_modes) as f64);
    let qz = omega_0 / C;

    // Define photon modes
    let qx_values = Array1::linspace(-0.0008, 0.0008, n_modes); // Photon wavevectors along x
    let qy_values = Array1::linspace(-0.0008, 0.0008, n_modes);

    let q_values: Vec<Vector3D> = qx_values.iter().flat_map(|qx| {
        qy_values.iter().map(|qy| {
            Vector3D { x: *qx, y: *qy, z: qz }
        })}).collect();

    // // 1D Array of q in x-direction
    // let qx_values = Array1::linspace(-0.001, 0.001, n_modes*n_modes);

    // let q_values: Vec<Vector3D> = qx_values.iter().map(|qx| {
    //         Vector3D { x: *qx, y: 0.0, z: qz }
    //     }).collect();
    
    // Define phonon parameters
    // let pn_coupling = 0.01; // 114 cm^-1
    // let pn_coupling = 0.004385964912; // 50 cm^-1
    // let pn_coupling = 0.008771929825; // 100 cm^-1
    // let pn_coupling = 0.0;
    let pn_omega = 0.0025; // 500 cm^-1
    // let pn_coupling = 0.001; // 11.4 cm^-1
    let pn_coupling = 0.0005; // 11.4 cm^-1
    let pn_k = arr1(&[0.,0.05]); // 1 nm^1
    let pn_n_fock = 1;

    // Parameters for dielectric function calculation
    let n_eig = 2048;  // Number of eigenstates to use
    let omegas = Array1::linspace(0.0, 0.5, 100000);  // Frequency range
    let gamma = 1e-4;  // Broadening parameter ~0.9 meV

    let omega_range;
    if !zoomed {
        omega_range = Some((0.172, 0.202));
    }
    else{
        omega_range = Some((0.176,0.178));
    }

    // Create directory for saving results
    let save_dir = "dielectric_scan";
    std::fs::create_dir_all(save_dir).unwrap_or_else(|_| {
        timed_println!("Directory {} already exists or couldn't be created", save_dir);
    });


    // Create more descriptive filenames
    let mut real_filename = format!("dielectric_real_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}", 
                              e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock);
    let mut imag_filename = format!("dielectric_imag_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}", 
                              e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock);
    let mut both_filename = format!("dielectric_both_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}", 
                              e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock);
    if !zoomed {
        real_filename += ".svg";
        imag_filename += ".svg";
        both_filename += ".svg";
    }
    else{
        real_filename += "_zoomed.svg";
        imag_filename += "_zoomed.svg";
        both_filename += "_zoomed.svg";
    }

    if !calculate {
        // Just plot the existing data
        timed_println!("Plotting dielectric function from existing data...");
        
        // Create reference parameters to match against
        let ref_params = visualization::DielectricParams {
            a0: e_config.a0,
            v0: e_config.v0,
            n_rel,
            lm_coupling: 0.0, // This will be matched separately using coupling_values
            n_modes: n_modes*n_modes,
            pn_coupling,
            pn_n_fock,
            n_eig,
            gamma,
        };
        
        match plot_dielectric_function(save_dir, Some(&real_filename), DielectricPlotType::Real, omega_range, &coupling_values, Some(&ref_params), zoomed, Some(&q_values)) {
            Ok(_) => timed_println!("Successfully plotted real part of dielectric function"),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }
        
        match plot_dielectric_function(save_dir, Some(&imag_filename), DielectricPlotType::Imaginary, omega_range, &coupling_values, Some(&ref_params), zoomed, None) {
            Ok(_) => timed_println!("Successfully plotted imaginary part of dielectric function"),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }
        
        match plot_dielectric_function(save_dir, Some(&both_filename), DielectricPlotType::Both, omega_range, &coupling_values, Some(&ref_params), zoomed, Some(&q_values)) {
            Ok(_) => timed_println!("Successfully plotted both parts of dielectric function"),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }
        
        return;
    }

    timed_println!("Starting dielectric function scan for {} coupling values", coupling_values.len());
    
    // Get MPI rank and size from environment variables
    let world_size: usize = env::var("PMI_SIZE")
        .or_else(|_| env::var("PMIX_SIZE"))
        .or_else(|_| env::var("I_MPI_SIZE"))
        .or_else(|_| env::var("OMPI_COMM_WORLD_SIZE"))
        .or_else(|_| env::var("SLURM_NTASKS"))
        .unwrap_or_else(|_| {
            timed_println!("Could not find MPI world size in environment variables. Using default value of 1.");
            1.to_string()
        })
        .parse()
        .expect("Failed to parse world size as integer");
    
    let rank: usize = env::var("PMI_RANK")
        .or_else(|_| env::var("PMIX_RANK"))
        .or_else(|_| env::var("I_MPI_RANK"))
        .or_else(|_| env::var("OMPI_COMM_WORLD_RANK"))
        .or_else(|_| env::var("SLURM_PROCID"))
        .unwrap_or_else(|_|{
            timed_println!("Could not find MPI rank in environment variables. Using default value of 0.");
            0.to_string()
        })
        .parse()
        .expect("Failed to parse rank as integer");

    if rank == 0 {
        timed_println!("Running with {} MPI ranks", world_size);
    }

    // Distribute coupling values across ranks
    let n_couplings = coupling_values.len();
    let couplings_per_rank = n_couplings / world_size;
    let start_idx = rank * couplings_per_rank;
    let end_idx = ((rank + 1) * couplings_per_rank).min(n_couplings);
    
    let my_couplings: Vec<f64> = coupling_values.slice(s![start_idx..end_idx]).to_owned().to_vec();

    // Create a unique temporary directory for this job
    let job_id = env::var("SLURM_JOB_ID")
        .or_else(|_| env::var("SLURM_JOBID"))
        .unwrap_or_else(|_| SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs().to_string());
    let temp_dir = format!("{}/mpi_temp_{}", save_dir, job_id);
    std::fs::create_dir_all(&temp_dir).unwrap_or_else(|_| {
        timed_println!("Failed to create temporary directory: {}", temp_dir);
        return;
    });

    // For each coupling strength assigned to this rank
    for (i, &coupling) in my_couplings.iter().enumerate() {
        let start_time = Instant::now();
        
        // Create PP config
        let pp_config = PPConfig{
            e_config: e_config.clone(), n_rel, n_ex_eig, lm_coupling: coupling, omega_0, q_values: q_values.clone(),
            pn_coupling, pn_omega, pn_k: pn_k.clone(), pn_n_fock, n_k_points: 1, n_bands: 1
        };


        // New Method
        timed_println!("Using new method");

        let mut pp: PPHamiltonian<ComposedOperator<c64>, SingExBosons<ComposedOperator<c64>>> = PPHamiltonian::from_config(&pp_config);
        let result = dielectric_function::get_dielectric_function(
            &pp_config, &mut pp, n_eig, omegas.clone(), gamma, save_dir, use_cached
        );
        let filename = dielectric_function::get_dielectric_function_filename(&pp, &pp_config, save_dir, n_eig, gamma);
        
        // // Generate the PP system
        // let mut pp_wrapper = sing_ex::SingExWrapper::new(pp_config.clone());
        // pp_wrapper.generate_pp();
        // let pp = pp_wrapper.polaron_polariton.as_ref().unwrap();
        
        // // Calculate dielectric function (will use cached diagonalization if available)
        // let (re_eps, im_eps) = sing_ex::get_dielectric_function(
        //     &pp_config, pp, n_eig, omegas.clone(), gamma, save_dir, use_cached
        // );
        
        // // Save the final dielectric function results
        // let filename = sing_ex::get_dielectric_function_filename(pp, &pp_config, save_dir, n_eig, gamma);

        timed_println!("Saving dielectric function data to {}", filename);
        let mut file = File::create(&filename).unwrap();
    
        // Write header
        writeln!(file, "# omega Re[eps] Im[eps] PhotonicChar PhononNumber").unwrap();
        
        // Write data
        for i in 0..omegas.len() {
            writeln!(file, "{:.6} {:.6} {:.6}", omegas[i], result.real[i], result.imag[i]).unwrap();
        }

        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);
        
        timed_println!("Completed coupling {:.3} ({}/{}) in {:?}", 
                          coupling, i+1, my_couplings.len(), duration);

    }
    
    // All ranks write barrier file after completing their work
    let barrier_file = format!("{}/rank_{}.done", temp_dir, rank);
    if let Err(e) = std::fs::write(&barrier_file, format!("{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs())) {
        timed_println!("Rank {}: Failed to write barrier file: {}", rank, e);
    } else {
        timed_println!("Rank {}: Wrote barrier file", rank);
    }

    // Gather results on rank 0
    if rank == 0 {
        // Wait for all ranks to finish
        timed_println!("Rank 0: Waiting for all ranks to finish...");
        loop {
            let mut completed_count = 0;
            for r in 0..world_size {
                let barrier_check = format!("{}/rank_{}.done", temp_dir, r);
                if Path::new(&barrier_check).exists() {
                    completed_count += 1;
                }
            }
            if completed_count == world_size {
                timed_println!("Rank 0: All {} ranks completed", world_size);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        timed_println!("Rank 0: All ranks finished, gathering results...");
   
        // Plot the results
        timed_println!("Plotting dielectric function...");
        timed_println!("Coupling values to plot: {:?}", coupling_values);
        
        // Make a copy of coupling_values to ensure we're using the exact same values
        let plot_coupling_values = coupling_values.clone();
        
        // Create reference parameters to match against
        let ref_params = visualization::DielectricParams {
            a0: e_config.a0,
            v0: e_config.v0,
            n_rel,
            lm_coupling: 0.0, // This will be matched separately using coupling_values
            n_modes: n_modes*n_modes,
            pn_coupling,
            pn_n_fock,
            n_eig,
            gamma,
        };
        
        match plot_dielectric_function(save_dir, Some(&real_filename), DielectricPlotType::Real, omega_range, &plot_coupling_values, Some(&ref_params), zoomed, Some(&q_values)) {
            Ok(_) => timed_println!("Successfully plotted real part of dielectric function to {}", real_filename),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }
        
        match plot_dielectric_function(save_dir, Some(&imag_filename), DielectricPlotType::Imaginary, omega_range, &plot_coupling_values, Some(&ref_params), zoomed, Some(&q_values)) {
            Ok(_) => timed_println!("Successfully plotted imaginary part of dielectric function to {}", imag_filename),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }
        
        match plot_dielectric_function(save_dir, Some(&both_filename), DielectricPlotType::Both, omega_range, &plot_coupling_values, Some(&ref_params), zoomed, Some(&q_values)) {
            Ok(_) => timed_println!("Successfully plotted both parts of dielectric function to {}", both_filename),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }

        // Remove temporary directory
        if let Err(e) = std::fs::remove_dir_all(&temp_dir) {
            timed_println!("Rank 0: Failed to remove temporary directory: {}", e);
        }
    }
}

fn main() {
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
        a0: 16.0,                                  // Lattice constant
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

    initialize().expect("Failed to initialize SLEPc");
    
    match run_mode {
        "bandstructure" => {
            // run_bandstructure(Method::LinOp, e_config.clone());
            // run_bandstructure(Method::SingExLegacy, e_config.clone());
            run_bandstructure(Method::Sparse, e_config.clone());
            // run_bandstructure(Method::Dense, e_config.clone());
        },
        "dielectric_scan" => {
            run_dielectric_scan(e_config.clone());
        },
        _ => {
            timed_println!("Unknown run mode: {}", run_mode);
        }
    }
    
    // Finalize SLEPc
    finalize().expect("Failed to finalize SLEPc");
}
