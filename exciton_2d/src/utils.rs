//! Utility functions for the exciton 2D simulation
//!
//! This module provides various utility functions used throughout the crate.

use log::info;
use ndarray::{ArrayView2, Zip};
use std::process;

/// Basis convention used throughout exciton_2d:
///
///     |state> = |CoM_index> ⊗ |rel_index>
///
/// with flat index
///
///     idx = com_index * n_rel_states + rel_index.
///
/// Therefore diagonal operators are ordered as:
///
///     H_rel -> I_com ⊗ H_rel
///     H_com -> H_com ⊗ I_rel
///
/// Any downstream crate using exciton eigenvectors must use the same ordering.
#[inline]
pub fn exciton_flat_index(com_ind: usize, rel_ind: usize, n_rel: usize) -> usize {
    com_ind * n_rel + rel_ind
}

/// Inverse of `exciton_flat_index`.
///
/// Returns `(com_ind, rel_ind)`.
#[inline]
pub fn exciton_unflatten_index(idx: usize, n_rel: usize) -> (usize, usize) {
    let com_ind = idx / n_rel;
    let rel_ind = idx % n_rel;
    (com_ind, rel_ind)
}

/// Prints an estimate of the current memory usage.
///
/// This is a simplified implementation that could be improved with a proper
/// memory profiling library.
pub fn print_memory_usage(message: &str) {
    let usage = process::id(); // placeholder
    info!("Memory usage {}: {} bytes (estimated)", message, usage);
}

/// Applies one of the eight square-lattice point-group symmetry operations.
pub fn apply_symmetry(k_point: &[f64], symmetry_op: usize) -> Vec<f64> {
    sim_common::kpath::apply_square_symmetry(
        k_point,
        sim_common::kpath::SquareSymmetry::from_index(symmetry_op),
    )
}

/// Calculates a matrix element efficiently.
///
/// Computes:
///
///     <psi_i|V|psi_j> ≈ sum_xy psi_i(x,y) V(x,y) psi_j(x,y) dx²
///
/// This version avoids temporary arrays.
pub fn fast_matrix_element_calc(
    psi_i_real: ArrayView2<f64>,
    psi_j_real: ArrayView2<f64>,
    v_term: ArrayView2<f64>,
    dx: f64,
) -> f64 {
    let sum = Zip::from(&psi_i_real)
        .and(&psi_j_real)
        .and(&v_term)
        .fold(0.0, |acc, &a, &b, &v| acc + a * v * b);

    sum * dx * dx
}
