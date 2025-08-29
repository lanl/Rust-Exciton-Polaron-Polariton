pub mod config;
pub mod grid;
pub mod utils;
pub mod matrix_elements;
pub mod hamiltonian;
pub mod band_structure;
pub mod visualization;
pub mod system;

// Re-export main components
pub use config::ExcitonConfig;
pub use utils::{print_memory_usage, apply_symmetry};
pub use grid::Grid;
pub use matrix_elements::MatrixElementCalculator;
pub use hamiltonian::{RelativeHamiltonian, FullHamiltonian};
pub use band_structure::BandStructureCalculator;
pub use visualization::{plot_rel_eigenstate, plot_spectrum, plot_band_structure};
pub use system::Exciton2DGridSystem;
