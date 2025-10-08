use exciton_2d::{Exciton2DGridSystem, ExcitonConfig};
use std::{f64::consts::PI, path::PathBuf};
use std::time::Instant;
use ndarray::arr1;

pub mod sparse_methods;

pub mod utils;
pub use utils::{Vector3D, PhotonMode, Polarization};

pub mod visualization;
pub use visualization::plot_band_structure;

use crate::utils::PPConfig;

pub mod sing_ex;

pub mod dense_methods;

// Speed of light in atomic units
const C: f64 = 137.036;

enum Method {
    Sparse,
    Dense,
    SingEx,
}

fn run_bandstructure(method: Method, e_config: ExcitonConfig) {
    
    // Create the 2D exciton system
    let a0 = (&e_config).a0;
    let v0 = (&e_config).v0;
    let exciton = Exciton2DGridSystem::from_config(e_config.clone());
    
    // Parameters for diagonalization
    let n_rel = 16;      // Number of relative coordinate states
    let n_ex_eig = 32;  // Number of exciton eigenstates
    let parity = 0;      // Parity of states to consider
    
    // Light-matter coupling parameters
    let coupling = 0.01;  // Coupling strength
    let omega_0 = 0.25;    // Photon frequency
    
    // Define photon modes
    let qx_values = vec![0.0, 0.01, 0.02]; // Photon wavevectors along x
    let qz = omega_0 / C;
    let n_fock = 2;      // Dimension of Fock space

    let n_k_points = 300;  // Number of k-points along the path
    let n_bands = 64;

    let q_values: Vec<Vector3D> = qx_values.iter().map(|qx| {
            Vector3D { x: *qx, y: 0.0, z: qz }
        }).collect();
    

    // Define phonon parameters
    let pn_coupling = 0.01;
    let pn_omega = 0.05;
    let pn_k = arr1(&[0.,0.5]);
    let pn_n_fock = 10;

    let pp_config = PPConfig{
        e_config, n_rel, n_ex_eig, lm_coupling: coupling, omega_0, q_values,
        pn_coupling, pn_omega, pn_k, pn_n_fock, n_k_points, n_bands
    };

    // Create photon modes
    let photon_modes: Vec<PhotonMode> = qx_values.iter().map(|qx| {
            PhotonMode::new(Vector3D { x: *qx, y: 0.0, z: qz }, Polarization::TE, n_fock)
        }).collect();

    match method {
        Method::Sparse => {

            println!("Using sparse methods for light-matter coupling.");

            // Create light-matter system
            let light_matter = sparse_methods::LightMatter::new(exciton.full_ham, photon_modes, coupling,
                                            n_rel, n_ex_eig, parity);

            // Define high-symmetry points in k-space for band structure calculation
            // Here we use Γ to X path (center to edge of Brillouin zone)
            let sym_points = vec![ vec![0.0,0.0], vec![PI / a0, 0.0], vec![PI / a0, PI / a0]];
            let sym_labels = vec!["Γ".to_string(), "X".to_string(), "M".to_string()];

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
            println!("Calc dispersion in {:?}", duration);

            // Plot the band structure
            plot_band_structure(
                &k_points, sym_points, sym_labels, &eigvals_list, n_bands, a0, 
                Some("disp_plot.png"), false, v0, coupling);

        },
        Method::Dense => {
            
            println!("Using dense methods for light-matter coupling.");

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
            println!("Calc dispersion in {:?}", duration);

            // Plot the band structure
            plot_band_structure(
                &k_points, sym_points, sym_labels, &eigvals_list, n_bands, a0, 
                Some("disp_plot.png"), false, v0, coupling);

        },   

        Method::SingEx => {

            println!("Using Mat-Vec methods for light-matter coupling in single photon limit.");
            
            let start_time = Instant::now();

            let mut pp_wrapper = sing_ex::SingExWrapper::new(pp_config.clone());
            pp_wrapper.generate_pp();

            let time1 = Instant::now();
            let gen_time = time1.duration_since(start_time);
            println!("Precomputed matrix elements in {:?}", gen_time);

            let (e_vals, _) = sing_ex::diagonalize_pp_sing_ex(&pp_config, pp_wrapper.polaron_polariton.unwrap(), 100);

            let time2 = Instant::now();
            let diag_time = time2.duration_since(time1);
            println!("Diagonalized gamma point in {:?}", diag_time);

            println!("{:?}", e_vals);
        }     
    }
}

/// Main function demonstrating the use of the light-matter coupling framework
fn main() {
    // Create exciton configuration
    let e_config = ExcitonConfig {
        n_kappa: 7,                               // Number of k-points in Brillouin zone
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
        n_jobs: 2,                                // Number of parallel jobs
        use_gpu: false,                           // Whether to use GPU acceleration
        batch_size: 32,                           // Batch size for calculations
    };
    // davidson::davidson_test();

    run_bandstructure(Method::SingEx, e_config.clone());
    // run_bandstructure(Method::Sparse, e_config.clone());
    // run_bandstructure(Method::Dense, e_config.clone());
    
    // // Define phonon modes for polaron-polariton calculation
    // let phonon_freq = 0.01;  // Phonon frequency in a.u.
    // let phonon_q = phonon_freq / C;
    // let phonon_modes: Vec<PhotonMode> = vec![
    //     PhotonMode::new(Vector3D { x: phonon_q, y: 0.0, z: 0.0 }, Polarization::TE, 2),
    //     PhotonMode::new(Vector3D { x: 0.0, y: phonon_q, z: 0.0 }, Polarization::TE, 2)
    // ];
    
    // // Phonon coupling strengths
    // let phonon_coupling = vec![0.005, 0.005];
    
    // // Create polaron-polariton system
    // let polaron_polariton = PolaronPolariton::new(
    //     phonon_modes, light_matter, phonon_coupling);
    
    // // Calculate polaron-polariton eigenvalues at Γ point
    // let start_time = Instant::now();
    // let (eigvals, _) = polaron_polariton.diagonalize_pp_ham_sparse(20);
    // let end_time = Instant::now();
    
    // // Print results
    // println!("Polaron-polariton eigenvalues at Γ point:");
    // println!("{:?}", eigvals);
    // println!("Calculation time: {:?}", end_time.duration_since(start_time));
}
