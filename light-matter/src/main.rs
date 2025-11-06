use exciton_2d::{Exciton2DGridSystem, ExcitonConfig};
use std::{f64::consts::PI, path::PathBuf, io::Write};
use std::time::Instant;
use ndarray::{arr1, Array1};
use rust_slepc_bindings::{initialize, finalize};


pub mod sparse_methods;

pub mod utils;
pub use utils::{Vector3D, PhotonMode, Polarization};

pub mod visualization;
pub use visualization::{plot_band_structure, plot_dielectric_function, DielectricPlotType};

pub mod color_band_plot;

use crate::utils::PPConfig;

pub mod sing_ex;

pub mod dense_methods;

pub mod davidson;

pub mod lobpcg;

pub mod slepc;

mod logging;

use crate::logging::init_logging;

// Speed of light in atomic units
const C: f64 = 137.036;

pub enum Method {
    Sparse,
    Dense,
    SingEx,
}

fn run_bandstructure(method: Method, e_config: ExcitonConfig) {

    // Create the 2D exciton system
    let a0 = (&e_config).a0;
    let v0 = (&e_config).v0;
    timed_println!("v0 = {}", v0);
    let exciton = Exciton2DGridSystem::from_config(e_config.clone());
    
    // Parameters for diagonalization
    let n_rel = 9;      // Number of relative coordinate states
    let n_ex_eig = 16;  // Number of exciton eigenstates
    let parity = 0;      // Parity of states to consider
    
    // Light-matter coupling parameters
    let total_coupling = 0.04;  // Coupling strength
    let omega_0 = 0.2;    // Photon frequency
    let n_modes = 16;
    let coupling = total_coupling / (n_modes as f64);

    // Define photon modes
    // let qx_values = vec![0.0]; // Photon wavevectors along x
    let qx_values = Array1::linspace(-0.001, 0.001, n_modes); // Photon wavevectors along x
    let qy_values = Array1::linspace(-0.001, 0.001, n_modes);
    let qz = omega_0 / C;
    let n_fock = 2;      // Dimension of Fock space

    let n_k_points = 1000;  // Number of k-points along the path
    let n_bands = 16;

    let q_values: Vec<Vector3D> = qx_values.iter().flat_map(|qx| {
        qy_values.iter().map(|qy| {
            Vector3D { x: *qx, y: *qy, z: qz }
        })}).collect();
    

    // Define phonon parameters
    let pn_coupling = 0.00;
    let pn_omega = 0.01;
    let pn_k = arr1(&[0.,0.1]);
    let pn_n_fock = 1;

    let pp_config = PPConfig{
        e_config, n_rel, n_ex_eig, lm_coupling: coupling, omega_0, q_values,
        pn_coupling, pn_omega, pn_k, pn_n_fock, n_k_points, n_bands
    };

    // Create photon modes
    let photon_modes: Vec<PhotonMode> = qx_values.iter().flat_map(|qx| {
            qy_values.iter().map( |qy| {
                PhotonMode::new(Vector3D { x: *qx, y: *qy, z: qz }, Polarization::TE, n_fock).clone()
            })
        }).collect();

    // Define high-symmetry points in k-space for band structure calculation
    let sym_points = vec![ vec![0.0,0.0], vec![PI / a0, 0.0], vec![PI / a0, PI / a0]];
    let sym_labels = vec!["Γ".to_string(), "X".to_string(), "M".to_string()];
    
    timed_println!("Running with {} modes", photon_modes.len());

    match method {
        Method::Sparse => {

            timed_println!("Using sparse methods for light-matter coupling.");

            // Create light-matter system
            let light_matter = sparse_methods::LightMatter::new(exciton.full_ham, photon_modes, coupling,
                                            n_rel, n_ex_eig, parity);

            // Start timing the band structure calculation
            let start_time = Instant::now();
            
            // Create band structure calculator
            let band_structure = sparse_methods::BandStructureCalculatorLM::new(
                light_matter, a0, v0, "lm_data".to_string(), sym_points.clone());
            
            let n_bands_calc = (n_bands as f64 * 1.1).ceil() as usize; // Calculate a few extra bands to ensure accuracy

            // Using sparse matrices and ARPACK to calculate eigenvalues
            let (k_points, eigvals_list) = band_structure.calculate_lm_band_structure_sparse(n_k_points, n_bands_calc);
            
            // End timing and print duration
            let end_time = Instant::now();
            let duration = end_time.duration_since(start_time);
            timed_println!("Calc dispersion in {:?}", duration);

            // Plot the band structure
            plot_band_structure(
                &k_points, sym_points, sym_labels, &eigvals_list, n_bands, a0, 
                Some("disp_plot.png"), false, v0, coupling);

        },
        Method::Dense => {
            
            timed_println!("Using dense methods for light-matter coupling.");

            // Create light-matter system
            let light_matter = dense_methods::LightMatter::new(exciton.full_ham, photon_modes, coupling,
                                            n_rel, n_ex_eig, parity);

            // Define high-symmetry points in k-space for band structure calculation
            // Here we use Γ to X path (center to edge of Brillouin zone)
            let sym_points = vec![ vec![0.0,0.0], vec![PI / a0, 0.0], vec![PI / a0, PI / a0]];
            let sym_labels = vec!["Γ".to_string(), "X".to_string(), "M".to_string()];

            // Start timing the band structure calculation
            let start_time = Instant::now();
            
            // Create band structure calculator
            let band_structure = dense_methods::BandStructureCalculatorLM::new(
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

        Method::SingEx => {
            timed_println!("Using Mat-Vec methods for light-matter coupling in single photon limit.");
            let n_eig = 512;
            timed_println!("Calculating {} eigenstates", n_eig);
            
            let start_time = Instant::now();

            let mut pp_wrapper = sing_ex::SingExWrapper::new(pp_config.clone());
            pp_wrapper.generate_pp();
            let mut pp = pp_wrapper.polaron_polariton.unwrap();

            let time1 = Instant::now();
            let gen_time = time1.duration_since(start_time);
            timed_println!("Precomputed matrix elements in {:?}", gen_time);
            
            // Choose which diagonalization method to use
            let diag_method = sing_ex::DiagonalizationMethod::SLEPc;
            
            // // Benchmark different methods
            // timed_println!("Using {:?} diagonalization method", diag_method);
            
            // // Diagonalize at gamma point
            // let (_, _) = sing_ex::diagonalize_pp_sing_ex(&pp, n_eig, &diag_method, None);

            // let time2 = Instant::now();
            // let diag_time = time2.duration_since(time1);
            // timed_println!("Diagonalized gamma point in {:?}", diag_time);

            // timed_println!("Eigvals: {:?}", eig_vals);
            // let omegas = Array1::linspace(-0.3, 0.0, 1000);
            // let gamma = 0.01;
            // let _ = sing_ex::get_dielectric_function(&pp_config, &pp, n_eig, omegas, gamma, &diag_method);

            // let time3 = Instant::now();
            // let dielec_time = time3.duration_since(time2);
            // timed_println!("Calculated Dielectric Function in {:?}", dielec_time);
            
            // // Calculate band structure
            // let (k_points, eigvals_list) = sing_ex::calculate_band_structure_sing_ex(
            //     &mut pp, n_k_points, n_eig, &sym_points, &pp_config, diag_method);

            // // Plot the band structure
            // plot_band_structure(
            //     &k_points, sym_points, sym_labels, &eigvals_list, n_bands, a0, 
            //     Some("sing_ex_disp_plot.png"), false, v0, coupling);
                
            // Calculate and plot color-coded band structure
            timed_println!("Calculating color-coded band structure...");
            let (k_points_color, eigvals_list_color, photonic_character_list) = 
                color_band_plot::calculate_color_band_structure(
                    &mut pp, n_k_points, n_eig, &sym_points, &pp_config, diag_method, true); // Use cached data if available, sort bands for continuity

            timed_println!("Calculated Band Structure in {:?}", Instant::now().duration_since(time1));
                
            // Plot using gnuplot
            if let Err(e) = color_band_plot::plot_color_band_structure(
                &k_points_color, &sym_points, &sym_labels, &eigvals_list_color,
                &photonic_character_list, n_eig, a0, "sing_ex_color_disp_plot.png", v0, coupling) {
                timed_println!("Error plotting color-coded band structure: {}", e);
            }
        } 
    }
}

/// Main function demonstrating the use of the light-matter coupling framework
/// Calculate dielectric function for a range of coupling strengths
fn run_dielectric_scan(e_config: ExcitonConfig) {
    // Check if we should calculate or just plot
    let calculate = true; // Set to false to skip calculation and just plot
    let use_cached = false; // Set to true to use cached results if available
    // Parameters for diagonalization
    let n_rel = 9;      // Number of relative coordinate states
    let n_ex_eig = 16;  // Number of exciton eigenstates
    
    // Light-matter coupling parameters
    // let coupling_values = Array1::linspace(0.00, 0.05, 1); // Scan from weak to strong coupling
    let total_coupling_values = Array1::from_vec(vec![0.0, 0.04, 0.06]); // Scan from weak to strong coupling
    let omega_0 = 0.2;    // Photon frequency
    let n_modes = 16;
    let coupling_values = total_coupling_values /( n_modes as f64);
    
    // Define photon modes
    let qx_values = Array1::linspace(-0.001, 0.001, n_modes); // Photon wavevectors along x
    let qy_values = Array1::linspace(-0.001, 0.001, n_modes);
    let qz = omega_0 / C;

    let q_values: Vec<Vector3D> = qx_values.iter().flat_map(|qx| {
        qy_values.iter().map(|qy| {
            Vector3D { x: *qx, y: *qy, z: qz }
        })}).collect();
    
    // Define phonon parameters
    let pn_coupling = 0.0005;
    let pn_omega = 0.0075; // 1500 cm^-1
    let pn_k = arr1(&[0.,0.05]); // 1 nm^1
    let pn_n_fock = 16;

    // Parameters for dielectric function calculation
    let n_eig = 2048+128;  // Number of eigenstates to use
    let omegas = Array1::linspace(0.1, 0.25, 10000);  // Frequency range
    let gamma = 1e-5;  // Broadening parameter
    let diag_method = sing_ex::DiagonalizationMethod::SLEPc;

    let omega_range = Some((0.16,0.24));

    // Create directory for saving results
    let save_dir = "dielectric_scan";
    std::fs::create_dir_all(save_dir).unwrap_or_else(|_| {
        timed_println!("Directory {} already exists or couldn't be created", save_dir);
    });
    
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
        
        match plot_dielectric_function(save_dir, Some("dielectric_function_real.svg"), DielectricPlotType::Real, omega_range, &coupling_values, Some(&ref_params)) {
            Ok(_) => timed_println!("Successfully plotted real part of dielectric function"),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }
        
        match plot_dielectric_function(save_dir, Some("dielectric_function_imag.svg"), DielectricPlotType::Imaginary, omega_range, &coupling_values, Some(&ref_params)) {
            Ok(_) => timed_println!("Successfully plotted imaginary part of dielectric function"),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }
        
        match plot_dielectric_function(save_dir, Some("dielectric_function_both.svg"), DielectricPlotType::Both, omega_range, &coupling_values, Some(&ref_params)) {
            Ok(_) => timed_println!("Successfully plotted both parts of dielectric function"),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }
        
        return;
    }

    timed_println!("Starting dielectric function scan for {} coupling values", coupling_values.len());
    
    // Create progress bar
    let pb = indicatif::ProgressBar::new(coupling_values.len() as u64);
    pb.set_style(indicatif::ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} couplings ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    // For each coupling strength
    for (i, &coupling) in coupling_values.iter().enumerate() {
        let start_time = Instant::now();
        
        // Create PP config
        let pp_config = PPConfig{
            e_config: e_config.clone(), n_rel, n_ex_eig, lm_coupling: coupling, omega_0, q_values: q_values.clone(),
            pn_coupling, pn_omega, pn_k: pn_k.clone(), pn_n_fock, n_k_points: 1, n_bands: 1
        };
        
        // Generate the filename first to check if it exists
        let mut pp_wrapper = sing_ex::SingExWrapper::new(pp_config.clone());
        pp_wrapper.generate_pp();
        let pp = pp_wrapper.polaron_polariton.as_ref().unwrap();
        let filename = sing_ex::get_dielectric_function_filename(pp, &pp_config, save_dir, n_eig, gamma);
        
        // Check if file already exists and we want to use cached results
        let (re_eps, im_eps) = if use_cached && std::path::Path::new(&filename).exists() {
            timed_println!("Using cached dielectric function data from {}", filename);
            let (omegas_loaded, re_eps_loaded, im_eps_loaded) = visualization::load_dielectric_data(std::path::Path::new(&filename)).unwrap();
            
            // Verify the loaded data has the expected length
            if omegas_loaded.len() == omegas.len() {
                (Array1::from_vec(re_eps_loaded), Array1::from_vec(im_eps_loaded))
            } else {
                timed_println!("Cached data has different length than requested, recalculating...");
                let (re, im) = sing_ex::get_dielectric_function(&pp_config, pp, n_eig, omegas.clone(), gamma, &diag_method);
                (re, im)
            }
        } else {
            // Calculate dielectric function
            sing_ex::get_dielectric_function(&pp_config, pp, n_eig, omegas.clone(), gamma, &diag_method)
        };
        
        // Save results to file if we calculated them
        if !use_cached || !std::path::Path::new(&filename).exists() {
            timed_println!("Saving dielectric function data to {}", filename);
            let mut file = std::fs::File::create(&filename).unwrap();
        
            // Write header
            writeln!(file, "# omega Re[eps] Im[eps]").unwrap();
            
            // Write data
            for i in 0..omegas.len() {
                writeln!(file, "{:.6} {:.6} {:.6}", omegas[i], re_eps[i], im_eps[i]).unwrap();
            }
        }

        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);
        
        pb.inc(1);
        pb.println(format!("Completed coupling {:.3} ({}/{}) in {:?}", 
                          coupling, i+1, coupling_values.len(), duration));
    }
    
    pb.finish_with_message("Dielectric function scan complete");
    
    // Plot the results
    timed_println!("Plotting dielectric function...");
    timed_println!("Coupling values to plot: {:?}", coupling_values);
    
    // Create more descriptive filenames
    let real_filename = format!("dielectric_real_a0{}_V0{}_nrel{}_nmodes{}_pncoup{:.3}_npn{}.svg", 
                              e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock);
    let imag_filename = format!("dielectric_imag_a0{}_V0{}_nrel{}_nmodes{}_pncoup{:.3}_npn{}.svg", 
                              e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock);
    let both_filename = format!("dielectric_both_a0{}_V0{}_nrel{}_nmodes{}_pncoup{:.3}_npn{}.svg", 
                              e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock);
    
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
    
    match plot_dielectric_function(save_dir, Some(&real_filename), DielectricPlotType::Real, omega_range, &plot_coupling_values, Some(&ref_params)) {
        Ok(_) => timed_println!("Successfully plotted real part of dielectric function to {}", real_filename),
        Err(e) => timed_println!("Error plotting dielectric function: {}", e),
    }
    
    match plot_dielectric_function(save_dir, Some(&imag_filename), DielectricPlotType::Imaginary, omega_range, &plot_coupling_values, Some(&ref_params)) {
        Ok(_) => timed_println!("Successfully plotted imaginary part of dielectric function to {}", imag_filename),
        Err(e) => timed_println!("Error plotting dielectric function: {}", e),
    }
    
    match plot_dielectric_function(save_dir, Some(&both_filename), DielectricPlotType::Both, omega_range, &plot_coupling_values, Some(&ref_params)) {
        Ok(_) => timed_println!("Successfully plotted both parts of dielectric function to {}", both_filename),
        Err(e) => timed_println!("Error plotting dielectric function: {}", e),
    }
}

fn main() {
    // Initialize logging
    init_logging();

    // Get the number of available cores
    let num_threads = num_cpus::get();
    // let num_threads = 16;

    // Set OpenBLAS thread count
    unsafe {
        // Set OpenBLAS thread count
        std::env::set_var("OPENBLAS_NUM_THREADS", num_threads.to_string());
        std::env::set_var("OMP_NUM_THREADS", num_threads.to_string());
        
        // Set MKL thread count (if using MKL)
        std::env::set_var("MKL_NUM_THREADS", num_threads.to_string());

        std::env::set_var("OMP_PROC_BIND", "spread");
        std::env::set_var("OMP_PLACES", "cores");
    }
    
    // Configure Rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();
    
    // Create exciton configuration
    let e_config = ExcitonConfig {
        n_kappa: 7,                               // Number of bands
        a0: 9.0,                                  // Lattice constant
        v0: 0.05,                                 // Potential strength
        k: vec![0.0, 0.0],                        // Initial k-vector
        m_e: 0.3,                                 // Electron effective mass
        m_h: 0.3,                                 // Hole effective mass
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
            run_bandstructure(Method::SingEx, e_config.clone());
            // run_bandstructure(Method::Sparse, e_config.clone());
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
