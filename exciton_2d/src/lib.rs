//! Exciton 2D - A library for simulating excitons in 2D materials
//!
//! This crate provides tools for calculating and visualizing exciton properties
//! in two-dimensional materials with periodic potentials. It implements a model
//! for excitons (bound electron-hole pairs) in a 2D periodic potential, allowing
//! for the calculation of energy levels, wavefunctions, and band structures.
//!
//! # Features
//!
//! - Calculation of exciton states in 2D materials
//! - Band structure calculations along high-symmetry paths
//! - Visualization of eigenstates and band structures
//! - Support for different effective masses and potential strengths
//! - Memory-efficient implementation for large grids
//!
//! # Example
//!
//! ```rust
//! use exciton_2d::{Exciton2DGridSystem, ExcitonConfig};
//! use std::path::PathBuf;
//!
//! // Create configuration
//! let config = ExcitonConfig {
//!     n_kappa: 11,
//!     a0: 3.0,
//!     v0: 0.05,
//!     k: vec![0.0, 0.0],
//!     m_e: 1.0,
//!     m_h: 1.0,
//!     verbose: true,
//!     grid_size: 64,
//!     box_size: 3.0,
//!     save_dir: PathBuf::from("saved_data"),
//!     recalculate: false,
//!     n_jobs: -1,
//!     use_gpu: false,
//!     batch_size: 32,
//! };
//!
//! // Create the 2D exciton system
//! let system = Exciton2DGridSystem::from_config(config);
//!
//! // Solve for the relative coordinate eigenstates
//! let (rel_eigvals, rel_eigvecs, rel_pars) = system.solve_rel_hamiltonian(16);
//!
//! // Calculate band structure
//! let (k_points, eigvals_list, parities_list) = system.calculate_band_structure(
//!     60,  // n_points
//!     9,   // n_rel_states
//!     16,  // n_bands
//!     true, // recalculate_bands
//!     0,    // select_parity (0: both, 1: even, -1: odd)
//! );
//! ```

pub mod band_structure;
pub mod config;
pub mod grid;
pub mod hamiltonian;
pub mod matrix_elements;
pub mod system;
pub mod utils;
pub mod visualization;

// Re-export main components for easier access
pub use band_structure::{
    BandStructureCalculator, ColorBandData, calculate_n_colored_band_structure,
    calculate_n_expectation,
};
pub use config::ExcitonConfig;
pub use grid::Grid;
pub use hamiltonian::{FullHamiltonian, RelativeHamiltonian};
pub use matrix_elements::MatrixElementCalculator;
pub use system::Exciton2DGridSystem;
pub use utils::{
    apply_symmetry, exciton_flat_index, exciton_unflatten_index, fast_matrix_element_calc,
    fast_matrix_element_calc_view, print_memory_usage,
};
pub use visualization::{plot_band_structure, plot_rel_eigenstate, plot_spectrum};
