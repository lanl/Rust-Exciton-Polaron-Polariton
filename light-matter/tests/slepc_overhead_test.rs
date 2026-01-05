// Overhead analysis tests for SLEPc integration

use exciton_2d::{Exciton2DGridSystem, ExcitonConfig};
use light_matter::Vector3D;
use light_matter::sing_ex::{PolaronPolariton, ExcitonicHamiltonian, PhotonicHamiltonian, PhononHamiltonian, SolverOperator};
use light_matter::utils::{PhotonMode, Polarization};
use light_matter::slepc_instrumented::{diagonalize_slepc_instrumented, get_instrumentation_data};
use ndarray::{Array1, arr1};
use std::sync::Arc;
use std::time::Instant;
use rust_slepc_bindings::EPSType;
use std::path::PathBuf;

// Speed of light in atomic units
const C: f64 = 137.036;

/// Test configuration for overhead measurements
struct OverheadTestConfig {
    n_ex_eig: usize,
    n_rel: usize,
    n_photon_modes: usize,
    n_phonon_fock: usize,
    n_eig_requested: usize,
    n_matvec_iterations: usize,
}

impl Default for OverheadTestConfig {
    fn default() -> Self {
        Self {
            n_ex_eig: 10,
            n_rel: 5,  // Reduced from 20 to avoid Laguerre polynomial order >7
            n_photon_modes: 2,
            n_phonon_fock: 3,
            n_eig_requested: 5,
            n_matvec_iterations: 1000,
        }
    }
}

/// Create a test PolaronPolariton system
fn create_test_system(config: &OverheadTestConfig) -> PolaronPolariton {
    // Create a minimal exciton system using ExcitonConfig
    let e_config = ExcitonConfig {
        n_kappa: 7,
        a0: 1.0,
        v0: 1.0,
        k: vec![0.0, 0.0],
        m_e: 1.0,
        m_h: 1.0,
        verbose: false,
        grid_size: 64,
        box_size: 3.0,
        save_dir: PathBuf::from("test_overhead_data"),
        recalculate: false,
        n_jobs: 1,
        use_gpu: false,
        batch_size: 32,
    };
    
    let ex_system = Exciton2DGridSystem::from_config(e_config);
    let mut exciton = ExcitonicHamiltonian::new(
        ex_system.full_ham,
        config.n_ex_eig,
        config.n_rel,
        0
    );
    exciton.compute_eigenstates();
    

    // Define photon modes
    let qx_values = Array1::linspace(-0.001, 0.001, config.n_photon_modes); // Photon wavevectors along x
    let qy_values = Array1::linspace(-0.001, 0.001, config.n_photon_modes);
    let qz = 0.15 / C;
    let n_fock = 2;      // Dimension of Fock space

    let q_values: Vec<Vector3D> = qx_values.iter().flat_map(|qx| {
        qy_values.iter().map(|qy| {
            Vector3D { x: *qx, y: *qy, z: qz }
        })}).collect();

    // Create photon modes
    let photon_modes: Vec<PhotonMode> = q_values.iter().map( |q| {
                PhotonMode::new(q.clone(), Polarization::TE, n_fock).clone()
        }).collect();
    let photon = PhotonicHamiltonian::new(photon_modes);
    
    // Create phonon mode if needed
    let phonon = if config.n_phonon_fock > 1 {
        let phonon_mode = PhotonMode {
            eps: arr1(&[0.0, 0.0]),
            q: arr1(&[0.,0.05]),
            omega: 0.0075,
            n_fock: config.n_phonon_fock,
        };
        Some(Arc::new(PhononHamiltonian::new(phonon_mode)))
    } else {
        None
    };
    
    PolaronPolariton::new(
        Arc::new(exciton),
        Arc::new(photon),
        phonon,
        vec![0.0, 0.0],
        0.015 / config.n_photon_modes as f64,  // lm_coupling
        0.00005, // pn_coupling
    )
}

/// Measure the time for pure matrix-vector products
fn benchmark_matvec(pp: &PolaronPolariton, n_iterations: usize) -> (f64, f64) {
    let dim = pp.dimension();
    let x = Array1::from_elem(dim, 1.0 / (dim as f64).sqrt());
    let mut y = Array1::zeros(dim);
    
    // Warm-up
    for _ in 0..10 {
        pp.apply(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
    }
    
    // Actual benchmark
    let start = Instant::now();
    for _ in 0..n_iterations {
        pp.apply(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
    }
    let duration = start.elapsed();
    
    let total_time = duration.as_secs_f64();
    let time_per_matvec = total_time / n_iterations as f64;
    
    (total_time, time_per_matvec)
}

/// Measure the time for SLEPc diagonalization
fn benchmark_slepc(
    pp: &PolaronPolariton,
    n_eig: usize,
    solver_type: EPSType,
    block_size: usize,
) -> (f64, usize, f64, usize, f64) {
    let start = Instant::now();
    let (_eigenvalues, _eigenvectors) = diagonalize_slepc_instrumented(
        pp,
        n_eig,
        block_size,
        Some(solver_type),
        None,
    );
    let duration = start.elapsed();
    
    let total_time = duration.as_secs_f64();
    let (matvec_count, matvec_total_time) = get_instrumentation_data();
    
    let estimated_iterations = matvec_count;
    let time_per_iteration = if estimated_iterations > 0 {
        total_time / estimated_iterations as f64
    } else {
        0.0
    };
    
    (total_time, estimated_iterations, time_per_iteration, matvec_count, matvec_total_time)
}

/// Calculate overhead metrics
struct OverheadMetrics {
    matvec_time: f64,
    slepc_total_time: f64,
    slepc_iterations: usize,
    slepc_time_per_iteration: f64,
    estimated_pure_matvec_time: f64,
    overhead_time: f64,
    overhead_percentage: f64,
    dimension: usize,
    actual_matvec_count: usize,
    actual_matvec_time: f64,
    callback_overhead: f64,
}

impl OverheadMetrics {
    fn print_report(&self, solver_name: &str) {
        println!("\n{}", "=".repeat(60));
        println!("Overhead Analysis for {}", solver_name);
        println!("{}", "=".repeat(60));
        println!("System dimension: {}", self.dimension);
        println!("\nMatrix-Vector Product Benchmark:");
        println!("  Time per matvec (isolated): {:.6} ms", self.matvec_time * 1000.0);
        println!("\nSLEPc Solver Performance:");
        println!("  Number of iterations: {}", self.slepc_iterations);
        println!("  Time per iteration: {:.3} s", self.slepc_time_per_iteration);
        println!("  Total time: {:.3} s", self.slepc_total_time);
        println!("  Matrix-vector calls: {}", self.actual_matvec_count);
        println!("  Time per matvec call: {:.6} ms", 
                 if self.actual_matvec_count > 0 { 
                     (self.slepc_total_time / self.actual_matvec_count as f64) * 1000.0 
                 } else { 
                     0.0 
                 });
        println!("\nCallback Overhead Analysis:");
        println!("  Actual matvec time in callbacks: {:.3} s", self.actual_matvec_time);
        println!("  Callback overhead (FFI + data transfer): {:.3} s ({:.2}%)", 
                 self.callback_overhead,
                 (self.callback_overhead / self.slepc_total_time) * 100.0);
        println!("  Callback overhead per call: {:.6} ms",
                 if self.actual_matvec_count > 0 {
                     (self.callback_overhead / self.actual_matvec_count as f64) * 1000.0
                 } else {
                     0.0
                 });
        println!("\nTotal Overhead Analysis:");
        println!("  Estimated pure matvec time: {:.3} s", self.estimated_pure_matvec_time);
        println!("  Total overhead time: {:.3} s", self.overhead_time);
        println!("  Total overhead percentage: {:.2}%", self.overhead_percentage);
        println!("  SLEPc algorithm overhead: {:.3} s ({:.2}%)",
                 self.overhead_time - self.callback_overhead,
                 ((self.overhead_time - self.callback_overhead) / self.slepc_total_time) * 100.0);
        println!("{}\n", "=".repeat(60));
    }
}

fn calculate_overhead(
    pp: &PolaronPolariton,
    config: &OverheadTestConfig,
    solver_type: EPSType,
    solver_name: &str,
    block_size: usize,
) -> OverheadMetrics {
    println!("\nTesting {} solver...", solver_name);
    
    println!("  Benchmarking matrix-vector products...");
    let (_, matvec_time) = benchmark_matvec(pp, config.n_matvec_iterations);
    
    println!("  Running SLEPc diagonalization...");
    let (slepc_total_time, slepc_iterations, slepc_time_per_iteration, actual_matvec_count, actual_matvec_time) = 
        benchmark_slepc(pp, config.n_eig_requested, solver_type, block_size);
    
    let estimated_pure_matvec_time = matvec_time * actual_matvec_count as f64;
    let overhead_time = slepc_total_time - estimated_pure_matvec_time;
    let overhead_percentage = (overhead_time / slepc_total_time) * 100.0;
    let callback_overhead = actual_matvec_time - estimated_pure_matvec_time;
    
    OverheadMetrics {
        matvec_time,
        slepc_total_time,
        slepc_iterations,
        slepc_time_per_iteration,
        estimated_pure_matvec_time,
        overhead_time,
        overhead_percentage,
        dimension: pp.dimension(),
        actual_matvec_count,
        actual_matvec_time,
        callback_overhead,
    }
}

#[test]
fn test_slepc_overhead_small_system() {
    {
        rust_slepc_bindings::initialize().expect("Failed to initialize SLEPc");
    }
    
    let config = OverheadTestConfig {
        n_ex_eig: 5,
        n_rel: 4,  // Reduced to avoid Laguerre polynomial order >7
        n_photon_modes: 5,
        n_phonon_fock: 2,
        n_eig_requested: 3,
        n_matvec_iterations: 1000,
    };
    
    println!("\n{}", "=".repeat(60));
    println!("SMALL SYSTEM TEST");
    println!("{}", "=".repeat(60));
    println!("Configuration:");
    println!("  n_ex_eig: {}", config.n_ex_eig);
    println!("  n_rel: {}", config.n_rel);
    println!("  n_photon_modes: {}", config.n_photon_modes);
    println!("  n_phonon_fock: {}", config.n_phonon_fock);
    println!("  n_eig_requested: {}", config.n_eig_requested);
    
    let pp = create_test_system(&config);
    println!("  Total dimension: {}", pp.dimension());
 
    let metrics = calculate_overhead(&pp, &config, EPSType::KrylovSchur, "Krylov-Schur", 128);
    metrics.print_report("Krylov-Schur");
    
    {
        rust_slepc_bindings::finalize().expect("Failed to finalize SLEPc");
    }
}

#[test]
fn test_slepc_overhead_medium_system() {
    {
        rust_slepc_bindings::initialize().expect("Failed to initialize SLEPc");
    }
    
    let config = OverheadTestConfig {
        n_ex_eig: 16,
        n_rel: 9,  // Reduced to avoid Laguerre polynomial order >7
        n_photon_modes: 11,
        n_phonon_fock: 4,
        n_eig_requested: 256,
        n_matvec_iterations: 1000,
    };
    
    println!("\n{}", "=".repeat(60));
    println!("MEDIUM SYSTEM TEST");
    println!("{}", "=".repeat(60));
    println!("Configuration:");
    println!("  n_ex_eig: {}", config.n_ex_eig);
    println!("  n_rel: {}", config.n_rel);
    println!("  n_photon_modes: {}", config.n_photon_modes);
    println!("  n_phonon_fock: {}", config.n_phonon_fock);
    println!("  n_eig_requested: {}", config.n_eig_requested);
    
    let pp = create_test_system(&config);
    println!("  Total dimension: {}", pp.dimension());
 
    let metrics = calculate_overhead(&pp, &config, EPSType::KrylovSchur, "Krylov-Schur", 128);
    metrics.print_report("Krylov-Schur");
    
    {
        rust_slepc_bindings::finalize().expect("Failed to finalize SLEPc");
    }
}

#[test]
fn test_slepc_overhead_large_system() {
    {
        rust_slepc_bindings::initialize().expect("Failed to initialize SLEPc");
    }
    
    let config = OverheadTestConfig {
        n_ex_eig: 16,
        n_rel: 9,
        n_photon_modes: 23,
        n_phonon_fock: 2,
        n_eig_requested: 2048,
        n_matvec_iterations: 1000,
    };
    
    println!("\n{}", "=".repeat(60));
    println!("LARGE SYSTEM TEST");
    println!("{}", "=".repeat(60));
    println!("Configuration:");
    println!("  n_ex_eig: {}", config.n_ex_eig);
    println!("  n_rel: {}", config.n_rel);
    println!("  n_photon_modes: {}", config.n_photon_modes);
    println!("  n_phonon_fock: {}", config.n_phonon_fock);
    println!("  n_eig_requested: {}", config.n_eig_requested);
    
    let pp = create_test_system(&config);
    println!("  Total dimension: {}", pp.dimension());
 
    let metrics = calculate_overhead(&pp, &config, EPSType::KrylovSchur, "Krylov-Schur", 128);
    metrics.print_report("Krylov-Schur");
    
    {
        rust_slepc_bindings::finalize().expect("Failed to finalize SLEPc");
    }
}