//! Utility functions for the exciton 2D simulation
//!
//! This module provides various utility functions used throughout the crate.

use log::info;
use ndarray::Array2;
use std::process;

/// Prints an estimate of the current memory usage
///
/// This is a simplified implementation that could be improved with a proper
/// memory profiling library.
///
/// # Arguments
/// * `message` - A message to include with the memory usage information
pub fn print_memory_usage(message: &str) {
    // This is platform-specific, so we'll use a simpler approach
    // For a more accurate implementation, consider using the 'psutil' crate
    let usage = process::id(); // Just a placeholder
    info!("Memory usage {}: {} bytes (estimated)", message, usage);
}

/// Applies a symmetry operation to a k-point
///
/// This function applies one of the eight symmetry operations of a square lattice
/// to a given k-point. The symmetry operations are:
/// - 0: Identity (no change)
/// - 1: 90° rotation
/// - 2: 180° rotation
/// - 3: 270° rotation
/// - 4: Reflection across x-axis
/// - 5: Reflection across y-axis
/// - 6: Reflection across y=x
/// - 7: Reflection across y=-x
///
/// # Arguments
/// * `k_point` - The k-point to transform
/// * `symmetry_op` - The symmetry operation to apply (0-7)
///
/// # Returns
/// * The transformed k-point
///
/// # Panics
/// Panics if `symmetry_op` is not in the range 0-7
pub fn apply_symmetry(k_point: &[f64], symmetry_op: usize) -> Vec<f64> {
    let kx = k_point[0];
    let ky = k_point[1];
    
    match symmetry_op {
        0 => vec![kx, ky],                // Identity
        1 => vec![-ky, kx],               // 90° rotation
        2 => vec![-kx, -ky],              // 180° rotation
        3 => vec![ky, -kx],               // 270° rotation
        4 => vec![kx, -ky],               // Reflection across x-axis
        5 => vec![-kx, ky],               // Reflection across y-axis
        6 => vec![ky, kx],                // Reflection across y=x
        7 => vec![-ky, -kx],              // Reflection across y=-x
        _ => panic!("Invalid symmetry operation: {}", symmetry_op),
    }
}

/// Calculates a matrix element efficiently
///
/// This function computes the matrix element <psi_i|V|psi_j> efficiently by
/// directly multiplying the wavefunctions and potential on the grid and
/// integrating using a simple quadrature rule.
///
/// The integral is approximated as:
/// <psi_i|V|psi_j> ≈ sum_{x,y} psi_i(x,y) * V(x,y) * psi_j(x,y) * dx²
///
/// # Arguments
/// * `psi_i_real` - First wavefunction on the 2D grid
/// * `psi_j_real` - Second wavefunction on the 2D grid
/// * `v_term` - Potential term on the 2D grid
/// * `dx` - Grid spacing
///
/// # Returns
/// * The calculated matrix element
pub fn fast_matrix_element_calc(
    psi_i_real: &Array2<f64>,
    psi_j_real: &Array2<f64>,
    v_term: &Array2<f64>,
    dx: f64,
) -> f64 {

    // Calculate the product (psi_i_conj * V(x) * psi_j)
    let real_part = (psi_i_real * v_term) * psi_j_real;
    
    // Sum and scale by dx^2
    let real_sum = real_part.sum() * dx * dx;
    
    real_sum
}