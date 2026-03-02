//! Demonstration of KronSum optimization for light-matter coupling
//!
//! This example shows how the KronSum optimization automatically detects
//! and optimizes the common pattern in light-matter Hamiltonians.

use ndarray::Array2;
use ndarray_linalg::c64;
use num_traits::{One, Zero};
use light_matter::utils::ComposedOperator;
use sprs::TriMat;
use std::time::Instant;

fn main() {
    println!("\n=== KronSum Optimization Demo ===");
    println!("\nThis demonstrates automatic optimization of light-matter coupling patterns.\n");
    
    // System parameters
    let n_exciton = 16;  // Exciton states
    let n_photon = 122;  // Photon modes (vacuum + 121 modes)
    let n_terms = 1000;  // Number of interaction terms
    
    println!("System configuration:");
    println!("  Exciton states: {}", n_exciton);
    println!("  Photon modes: {} (including vacuum)", n_photon);
    println!("  Interaction terms: {}", n_terms);
    println!("  Total Hilbert space dimension: {}\n", n_exciton * n_photon);
    
    // Create random exciton matrices
    println!("Creating {} random exciton matrices...", n_terms);
    let mut exciton_matrices = Vec::new();
    for _ in 0..n_terms {
        let mut m = Array2::zeros((n_exciton, n_exciton));
        for i in 0..n_exciton {
            for j in 0..n_exciton {
                m[[i, j]] = c64::new(
                    (i as f64 + j as f64) * 0.01,
                    (i as f64 - j as f64) * 0.005,
                );
            }
        }
        exciton_matrices.push(m);
    }
    
    // Create photon transition operators
    // Pattern: vacuum (state 0) -> photon mode (states 1..n_photon)
    println!("Creating photon transition operators...");
    let mut photon_ops = Vec::new();
    for i in 1..n_photon {
        let mut trimat = TriMat::new((n_photon, n_photon));
        trimat.add_triplet(i, 0, c64::one()); // |i⟩⟨0|
        photon_ops.push(trimat.to_csr());
    }
    
    // Build interaction terms: M_i ⊗ |photon_i⟩⟨vacuum|
    println!("\nBuilding {} Kronecker product terms...", n_terms);
    let start = Instant::now();
    let mut terms = Vec::new();
    for i in 0..n_terms {
        let photon_idx = (i % (n_photon - 1)) as usize;
        let exciton_op = ComposedOperator::from_operator(
            exciton_matrices[i].clone()
        );
        let photon_op = ComposedOperator::from_operator(
            photon_ops[photon_idx].clone()
        );
        terms.push(ComposedOperator::Kron(
            Box::new(exciton_op),
            Box::new(photon_op),
        ));
    }
    let build_time = start.elapsed();
    println!("  Built terms in {:?}", build_time);
    
    // Create sum with automatic optimization
    println!("\nCreating sum operator (automatic optimization)...");
    let start = Instant::now();
    let hamiltonian = ComposedOperator::sum(terms);
    let sum_time = start.elapsed();
    println!("  Sum created in {:?}", sum_time);
    
    // Check if optimization was applied
    if hamiltonian.is_kron_sum() {
        println!("\n✓ KronSum optimization APPLIED!");
        if let Some((n_transitions, ex_dim, ph_dim)) = hamiltonian.get_kron_sum_info() {
            println!("  Unique transitions: {}", n_transitions);
            println!("  Exciton dimension: {}", ex_dim);
            println!("  Photon dimension: {}", ph_dim);
            println!("  Compression ratio: {:.1}×", n_terms as f64 / n_transitions as f64);
        }
    } else {
        println!("\n✗ KronSum optimization NOT applied");
        println!("  Using standard Sum variant");
    }
    
    // Benchmark apply operation
    println!("\nBenchmarking operator application...");
    let total_dim = n_exciton * n_photon;
    let x: Vec<c64> = (0..total_dim)
        .map(|i| c64::new((i as f64 * 0.01).sin(), (i as f64 * 0.01).cos()))
        .collect();
    let mut y = vec![c64::zero(); total_dim];
    
    // Warm-up
    hamiltonian.apply(&x, &mut y);
    
    // Benchmark
    let n_iterations = 10;
    let start = Instant::now();
    for _ in 0..n_iterations {
        hamiltonian.apply(&x, &mut y);
    }
    let apply_time = start.elapsed();
    let avg_time = apply_time / n_iterations;
    
    println!("  Average apply time: {:?}", avg_time);
    println!("  Throughput: {:.2} GFLOP/s", 
        (2.0 * total_dim as f64 * total_dim as f64 / avg_time.as_secs_f64()) / 1e9);
    
    // Test diagonal extraction
    println!("\nExtracting diagonal...");
    let start = Instant::now();
    let diag = hamiltonian.get_diagonal();
    let diag_time = start.elapsed();
    println!("  Diagonal extracted in {:?}", diag_time);
    println!("  Diagonal size: {}", diag.len());
    
    // Summary
    println!("\n=== Summary ===");
    println!("Total construction time: {:?}", build_time + sum_time);
    println!("Average apply time: {:?}", avg_time);
    if hamiltonian.is_kron_sum() {
        if let Some((n_transitions, _, _)) = hamiltonian.get_kron_sum_info() {
            println!("Optimization: {} terms → {} transitions ({:.1}× reduction)",
                n_terms, n_transitions, n_terms as f64 / n_transitions as f64);
        }
    }
    println!("\nKronSum optimization demonstration complete!\n");
}
