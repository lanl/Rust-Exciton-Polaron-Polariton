//! A simple example of using the exciton_2d library
//!
//! This example creates an exciton system, solves for the relative coordinate
//! eigenstates, and calculates the band structure.

use exciton_2d::{Exciton2DGridSystem, ExcitonConfig};
use std::path::PathBuf;

fn main() {
    // Create configuration
    let config = ExcitonConfig {
        n_kappa: 11,
        a0: 3.0,
        v0: 0.05,
        k: vec![0.0, 0.0],
        m_e: 1.0,
        m_h: 1.0,
        verbose: true,
        grid_size: 64,
        box_size: 3.0,
        save_dir: PathBuf::from("saved_data"),
        recalculate: false,
        n_jobs: -1,
        use_gpu: false,
        batch_size: 32,
    };
    
    // Create the 2D exciton system
    let system = Exciton2DGridSystem::from_config(config);
    
    // Solve for the relative coordinate eigenstates
    let (rel_eigvals, rel_eigvecs, rel_pars) = system.solve_rel_hamiltonian(16);
    
    // Print the first few eigenvalues
    println!("Relative coordinate eigenvalues:");
    for i in 0..std::cmp::min(5, rel_eigvals.len()) {
        println!("E_{} = {:.6} with parity {}", i, rel_eigvals[i], rel_pars[i]);
    }
    
    // Plot the first eigenstate
    system.plot_rel_eigenstate(
        &rel_eigvecs.column(0).to_owned(),
        Some("Ground State"),
        Some("output_figures/ground_state.png")
    );
    
    // Calculate band structure
    let (k_points, eigvals_list) = system.calculate_band_structure(
        30,  // n_points
        9,   // n_rel_states
        16,  // n_bands
        true, // recalculate_bands
    );
    
    // Plot band structure
    system.plot_band_structure(
        &k_points,
        &eigvals_list,
        Some("output_figures/band_structure.png")
    );
    
    println!("Example completed successfully!");
}