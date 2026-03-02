//! Benchmark comparing sing_ex.rs implementation with ComposedOperator KronSum optimization
//!
//! This example builds the same polaron-polariton Hamiltonian using two methods:
//! 1. Hand-optimized sing_ex.rs implementation
//! 2. ComposedOperator with automatic KronSum optimization
//!
//! It then benchmarks the apply() operation for both methods to verify that
//! the KronSum optimization achieves similar performance to the hand-optimized code.

use ndarray::arr1;
use ndarray_linalg::c64;
use num_traits::Zero;
use std::time::Instant;


use light_matter::utils::{
    PPConfig, Vector3D, Polarization, BosonMode, 
    ComposedOperator, SolverOperator
};
use light_matter::hamiltonians::{
    LMHamiltonian, ExcitonicHamiltonian, BosonicHamiltonian, 
    SingExBosons
};
use light_matter::sing_ex::{
    SingExWrapper
};
use exciton_2d::{Exciton2DGridSystem, ExcitonConfig};

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  KronSum Optimization Benchmark vs Hand-Optimized sing_ex.rs   ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Configuration parameters
    let n_photon_modes = 121;
    let n_exciton_states = 16;
    let n_rel = 4;
    let lm_coupling = 0.01;
    let n_warmup = 5;
    let n_iterations = 50;

    println!("System Configuration:");
    println!("  Photon modes: {}", n_photon_modes);
    println!("  Exciton states: {}", n_exciton_states);
    println!("  Relative states: {}", n_rel);
    println!("  Light-matter coupling: {}", lm_coupling);
    println!("  Total Hilbert space dimension: {}\n", 
        n_exciton_states * (n_photon_modes + 1));

    // Create exciton configuration
    let e_config = ExcitonConfig {
        n_kappa: 5,
        a0: 1.0,
        v0: 0.1,
        k: vec![0.0, 0.0],
        m_e: 1.0,
        m_h: 1.0,
        verbose: false,
        grid_size: 64,
        box_size: 3.0,
        save_dir: std::path::PathBuf::from("saved_data_lm"),
        recalculate: false,
        n_jobs: 32,
        use_gpu: false,
        batch_size: 32,
    };

    // Create photon modes (uniform grid in k-space)
    println!("Creating {} photon modes...", n_photon_modes);
    let q_max = 0.5;
    let n_side = (n_photon_modes as f64).sqrt().ceil() as usize;
    let mut q_values = Vec::new();
    for i in 0..n_side {
        for j in 0..n_side {
            if q_values.len() >= n_photon_modes {
                break;
            }
            let qx = -q_max + 2.0 * q_max * i as f64 / (n_side - 1) as f64;
            let qy = -q_max + 2.0 * q_max * j as f64 / (n_side - 1) as f64;
            q_values.push(Vector3D::new(qx, qy, 0.0));
        }
    }
    q_values.truncate(n_photon_modes);

    // Create PP configuration
    let pp_config = PPConfig::new(
        e_config.clone(),
        n_rel,
        n_exciton_states,
        lm_coupling,
        0.1, // omega_0 (not used in single excitation)
        q_values.clone(),
        0.0, // No phonon coupling for this benchmark
        0.0, // pn_omega
        arr1(&[0.0, 0.0]),
        1, // pn_n_fock = 1 (no phonons)
        100, // n_k_points
        10, // n_bands
    );

    println!("\n{}", "=".repeat(70));
    println!("Building Hamiltonians...");
    println!("{}\n", "=".repeat(70));

    // ========================================================================
    // Method 1: Hand-optimized sing_ex.rs implementation
    // ========================================================================
    println!("Method 1: Hand-optimized sing_ex.rs");
    println!("{}", "-".repeat(70));
    
    let start = Instant::now();
    let mut sing_ex_wrapper = SingExWrapper::new(pp_config.clone());
    sing_ex_wrapper.generate_pp();
    let pp_sing_ex = sing_ex_wrapper.polaron_polariton.as_ref().unwrap();
    let build_time_sing_ex = start.elapsed();
    
    let dim_sing_ex = pp_sing_ex.dimension();
    println!("  Build time: {:?}", build_time_sing_ex);
    println!("  Dimension: {}", dim_sing_ex);

    // ========================================================================
    // Method 2: ComposedOperator with KronSum optimization
    // ========================================================================
    println!("\nMethod 2: ComposedOperator with KronSum");
    println!("{}", "-".repeat(70));
    
    let start = Instant::now();
    
    // Create exciton system
    let exciton_grid = Exciton2DGridSystem::from_config(e_config.clone());
    let mut exciton: ExcitonicHamiltonian<ComposedOperator<c64>> = 
        ExcitonicHamiltonian::new(exciton_grid.full_ham, n_exciton_states, n_rel);
    exciton.compute_eigenstates();
    
    // Create photon system with SingExBosons
    let photon_modes: Vec<BosonMode> = q_values.iter().map(|q| {
        BosonMode::new(*q, Polarization::TE, 2)
    }).collect();
    let photon: BosonicHamiltonian<ComposedOperator<c64>, SingExBosons<ComposedOperator<c64>>> = 
        BosonicHamiltonian::new(photon_modes);
    
    // Create light-matter Hamiltonian
    let mut lm_hamiltonian = LMHamiltonian::new(exciton, photon, lm_coupling);
    let ham_composed = lm_hamiltonian.build_lm_hamiltonian(None);
    
    let build_time_composed = start.elapsed();
    let dim_composed = ham_composed.dimension();
    
    println!("  Build time: {:?}", build_time_composed);
    println!("  Dimension: {}", dim_composed);
    
    // Check if KronSum optimization was applied
    if ham_composed.is_kron_sum() {
        if let Some((n_trans, ex_dim, ph_dim)) = ham_composed.get_kron_sum_info() {
            println!("  ✓ KronSum optimization APPLIED");
            println!("    Unique transitions: {}", n_trans);
            println!("    Exciton dimension: {}", ex_dim);
            println!("    Photon dimension: {}", ph_dim);
            println!("    Compression ratio: {:.1}×", 
                (n_photon_modes * n_exciton_states) as f64 / n_trans as f64);
        }
    } else {
        println!("  ✗ KronSum optimization NOT applied");
    }

    // Verify dimensions match
    assert_eq!(dim_sing_ex, dim_composed, 
        "Dimensions must match: sing_ex={}, composed={}", dim_sing_ex, dim_composed);

    println!("\n{}", "=".repeat(70));
    println!("Benchmarking apply() operation");
    println!("{}\n", "=".repeat(70));

    // Create test vectors
    let x: Vec<c64> = (0..dim_sing_ex)
        .map(|i| c64::new((i as f64 * 0.01).sin(), (i as f64 * 0.01).cos()))
        .collect();
    let mut y_sing_ex = vec![c64::zero(); dim_sing_ex];
    let mut y_composed = vec![c64::zero(); dim_composed];

    // ========================================================================
    // Warm-up phase
    // ========================================================================
    println!("Warm-up phase ({} iterations)...", n_warmup);
    for _ in 0..n_warmup {
        pp_sing_ex.apply(&x, &mut y_sing_ex);
        ham_composed.apply(&x, &mut y_composed);
    }

    // ========================================================================
    // Benchmark sing_ex.rs
    // ========================================================================
    println!("\nBenchmarking sing_ex.rs ({} iterations)...", n_iterations);
    let start = Instant::now();
    for _ in 0..n_iterations {
        pp_sing_ex.apply(&x, &mut y_sing_ex);
    }
    let time_sing_ex = start.elapsed();
    let avg_time_sing_ex = time_sing_ex / n_iterations;

    // ========================================================================
    // Benchmark ComposedOperator
    // ========================================================================
    println!("Benchmarking ComposedOperator ({} iterations)...", n_iterations);
    let start = Instant::now();
    for _ in 0..n_iterations {
        ham_composed.apply(&x, &mut y_composed);
    }
    let time_composed = start.elapsed();
    let avg_time_composed = time_composed / n_iterations;

    // ========================================================================
    // Verify correctness
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("Verifying correctness...");
    println!("{}\n", "=".repeat(70));

    let mut max_diff: f64 = 0.0;
    let mut avg_diff = 0.0;
    let mut n_large_diff = 0;
    
    for i in 0..dim_sing_ex {
        let diff = (y_sing_ex[i] - y_composed[i]).norm();
        max_diff = max_diff.max(diff);
        avg_diff += diff;
        if diff > 1e-8 {
            n_large_diff += 1;
        }
    }
    avg_diff /= dim_sing_ex as f64;

    println!("  Maximum difference: {:.2e}", max_diff);
    println!("  Average difference: {:.2e}", avg_diff);
    println!("  Elements with diff > 1e-8: {}", n_large_diff);

    if max_diff < 1e-6 {
        println!("  ✓ Results match within tolerance");
    } else {
        println!("  ✗ WARNING: Large differences detected!");
    }

    // ========================================================================
    // Performance Summary
    // ========================================================================
    println!("\n╔{}╗", "═".repeat(68));
    println!("║{:^68}║", "Performance Summary");
    println!("╠{}╣", "═".repeat(68));
    
    println!("║ {:30} │ {:15} │ {:15} ║", "Metric", "sing_ex.rs", "ComposedOp");
    println!("╠{}╣", "─".repeat(68));
    
    println!("║ {:30} │ {:>13.2?} │ {:>13.2?} ║", 
        "Build time", build_time_sing_ex, build_time_composed);
    
    println!("║ {:30} │ {:>13.2?} │ {:>13.2?} ║", 
        "Avg apply() time", avg_time_sing_ex, avg_time_composed);
    
    let speedup = avg_time_sing_ex.as_secs_f64() / avg_time_composed.as_secs_f64();
    let speedup_str = if speedup > 1.0 {
        format!("{:.2}× faster", speedup)
    } else {
        format!("{:.2}× slower", 1.0 / speedup)
    };
    
    println!("║ {:30} │ {:>31} ║", "ComposedOp vs sing_ex", speedup_str);
    
    let flops_sing_ex = 2.0 * dim_sing_ex as f64 * dim_sing_ex as f64 / avg_time_sing_ex.as_secs_f64();
    let flops_composed = 2.0 * dim_composed as f64 * dim_composed as f64 / avg_time_composed.as_secs_f64();
    
    println!("║ {:30} │ {:>13.2} │ {:>13.2} ║", 
        "Throughput (GFLOP/s)", flops_sing_ex / 1e9, flops_composed / 1e9);
    
    println!("╚{}╝\n", "═".repeat(68));

    // ========================================================================
    // Detailed Analysis
    // ========================================================================
    println!("Detailed Analysis:");
    println!("{}", "-".repeat(70));
    
    if ham_composed.is_kron_sum() {
        if let Some((n_trans, _, _)) = ham_composed.get_kron_sum_info() {
            let theoretical_speedup = (n_photon_modes * n_exciton_states) as f64 / n_trans as f64;
            println!("  Theoretical speedup from compression: {:.1}×", theoretical_speedup);
            println!("  Actual speedup: {:.2}×", speedup);
            
            if speedup > 0.8 * theoretical_speedup {
                println!("  ✓ Achieving near-theoretical performance");
            } else if speedup > 0.5 * theoretical_speedup {
                println!("  ⚠ Moderate performance (50-80% of theoretical)");
            } else {
                println!("  ✗ Below expected performance (<50% of theoretical)");
            }
        }
    }

    println!("\n  Memory usage estimate:");
    let mem_sing_ex = dim_sing_ex * dim_sing_ex * 16; // Complex matrix
    let mem_composed = if let Some((n_trans, ex_dim, _)) = ham_composed.get_kron_sum_info() {
        n_trans * ex_dim * ex_dim * 16 // Compressed matrices
    } else {
        mem_sing_ex
    };
    
    println!("    sing_ex.rs: {:.2} MB (full matrix)", mem_sing_ex as f64 / 1e6);
    println!("    ComposedOp: {:.2} MB (compressed)", mem_composed as f64 / 1e6);
    println!("    Memory reduction: {:.1}×", mem_sing_ex as f64 / mem_composed as f64);

    // ========================================================================
    // Conclusion
    // ========================================================================
    println!("\n{}", "=".repeat(70));
    println!("Conclusion:");
    println!("{}\n", "=".repeat(70));

    if max_diff < 1e-6 && speedup > 0.5 {
        println!("✓ SUCCESS: KronSum optimization is working correctly!");
        println!("  - Results match hand-optimized implementation");
        println!("  - Performance is competitive ({:.2}× relative to sing_ex.rs)", speedup);
        println!("  - Automatic optimization requires no manual code changes");
    } else if max_diff < 1e-6 {
        println!("⚠ PARTIAL SUCCESS: Results are correct but performance needs improvement");
        println!("  - Consider profiling to identify bottlenecks");
    } else {
        println!("✗ FAILURE: Results do not match!");
        println!("  - There may be a bug in the KronSum implementation");
    }

    println!("\nBenchmark complete!\n");
}