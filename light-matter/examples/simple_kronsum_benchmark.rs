//! Simple benchmark comparing sing_ex.rs with ComposedOperator KronSum
//!
//! This is a minimal example that demonstrates the KronSum optimization
//! with a small system that runs quickly.

use ndarray::{Array2, arr1};
use ndarray_linalg::c64;
use num_traits::Zero;
use sprs::CsMat;
use std::time::Instant;


use light_matter::utils::{
    BosonMode, ComposedOperator, PPConfig, Polarization, SolverOperator, Vector3D
};
use light_matter::hamiltonians::{
    LMHamiltonian, ExcitonicHamiltonian, BosonicHamiltonian, 
    SingExBosons
};
use light_matter::sing_ex::{
    SingExWrapper
};
use exciton_2d::ExcitonConfig;
use exciton_2d::Exciton2DGridSystem;

fn main() {
    println!("\n=== Simple KronSum Benchmark ===");
    println!("\nComparing hand-optimized sing_ex.rs with ComposedOperator KronSum\n");

    // Small system for quick testing
    let n_photon_modes = 5; // 5×5 grid
    let n_exciton_states = 8;
    let n_rel = 9;
    let lm_coupling = 0.1;
    
    let lm_couplings = vec![lm_coupling; n_photon_modes];

    println!("System: {} photon modes, {} exciton states", n_photon_modes, n_exciton_states);
    println!("Total dimension: {}\n", n_exciton_states * (n_photon_modes + 1));

    // Create configuration
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

    // Create photon modes
    let q_max = 0.3;
    let n_side = n_photon_modes;
    let mut q_values = Vec::new();
    for i in 0..n_side {
        for j in 0..n_side {
            let qx = -q_max + 2.0 * q_max * i as f64 / (n_side - 1) as f64;
            let qy = -q_max + 2.0 * q_max * j as f64 / (n_side - 1) as f64;
            q_values.push(Vector3D::new(qx, qy, 0.0));
        }
    }

    let pp_config = PPConfig::new(
        e_config.clone(),
        n_rel,
        n_exciton_states,
        lm_couplings.clone(),
        0.1,
        q_values.clone(),
        0.0, // No phonons
        0.0,
        arr1(&[0.0, 0.0]),
        1,
        100,
        10,
    );

    // Build sing_ex.rs version
    println!("Building sing_ex.rs Hamiltonian...");
    let start = Instant::now();
    let mut sing_ex_wrapper = SingExWrapper::new(pp_config.clone());
    sing_ex_wrapper.generate_pp();
    let pp_sing_ex = sing_ex_wrapper.polaron_polariton.as_ref().unwrap();
    println!("  Time: {:?}\n", start.elapsed());

    // Build dense version
    println!("Building Dense Hamiltonian...");
    let start = Instant::now();
    
    let exciton_grid = Exciton2DGridSystem::from_config(e_config.clone());
    let mut exciton: ExcitonicHamiltonian<Array2<c64>> = 
        ExcitonicHamiltonian::new(exciton_grid.full_ham, n_exciton_states, n_rel);
    exciton.compute_eigenstates();
    
    let photon_modes: Vec<BosonMode> = q_values.iter().map(|q| {
        BosonMode::new(*q, Polarization::TE, 2)
    }).collect();
    let photon: BosonicHamiltonian<Array2<c64>, SingExBosons<Array2<c64>>> = 
        BosonicHamiltonian::new(photon_modes);
    
    let mut lm_hamiltonian = LMHamiltonian::new(exciton, photon, lm_couplings.clone());
    let ham_dense = lm_hamiltonian.build_lm_hamiltonian(None);
    println!("  Time: {:?}", start.elapsed());

    // Build sparse version
    println!("Building Sparse Hamiltonian...");
    let start = Instant::now();
    
    let exciton_grid = Exciton2DGridSystem::from_config(e_config.clone());
    let mut exciton: ExcitonicHamiltonian<CsMat<c64>> = 
        ExcitonicHamiltonian::new(exciton_grid.full_ham, n_exciton_states, n_rel);
    exciton.compute_eigenstates();
    
    let photon_modes: Vec<BosonMode> = q_values.iter().map(|q| {
        BosonMode::new(*q, Polarization::TE, 2)
    }).collect();
    let photon: BosonicHamiltonian<CsMat<c64>, SingExBosons<CsMat<c64>>> = 
        BosonicHamiltonian::new(photon_modes);
    
    let mut lm_hamiltonian = LMHamiltonian::new(exciton, photon, lm_couplings.clone());
    let ham_sparse = lm_hamiltonian.build_lm_hamiltonian(None);
    println!("  Time: {:?}", start.elapsed());

    // Build ComposedOperator version
    println!("Building ComposedOperator Hamiltonian...");
    let start = Instant::now();
    
    let exciton_grid = Exciton2DGridSystem::from_config(e_config.clone());
    let mut exciton: ExcitonicHamiltonian<ComposedOperator<c64>> = 
        ExcitonicHamiltonian::new(exciton_grid.full_ham, n_exciton_states, n_rel);
    exciton.compute_eigenstates();
    
    let photon_modes: Vec<BosonMode> = q_values.iter().map(|q| {
        BosonMode::new(*q, Polarization::TE, 2)
    }).collect();
    let photon: BosonicHamiltonian<ComposedOperator<c64>, SingExBosons<ComposedOperator<c64>>> = 
        BosonicHamiltonian::new(photon_modes);
    
    let mut lm_hamiltonian = LMHamiltonian::new(exciton, photon, lm_couplings.clone());
    let ham_composed = lm_hamiltonian.build_lm_hamiltonian(None);
    println!("  Time: {:?}", start.elapsed());

    // Check optimization status
    let is_pure_kronsum = match &ham_composed {
        light_matter::utils::ComposedOperator::KronSum { .. } => true,
        _ => false,
    };
    
    if is_pure_kronsum {
        if let Some((n_trans, ex_dim, ph_dim)) = ham_composed.get_kron_sum_info() {
            println!("  ✓ Pure KronSum: {} transitions ({}×{} blocks)\n", n_trans, ex_dim, ph_dim);
        }
    } else if ham_composed.is_kron_sum() {
        if let Some((n_trans, ex_dim, ph_dim)) = ham_composed.get_kron_sum_info() {
            println!("  ✓ Hybrid KronSum: {} transitions ({}×{} blocks)\n", n_trans, ex_dim, ph_dim);
        }
    } else {
        println!("  ✗ KronSum not applied\n");
    }

    // Benchmark apply() operation
    let dim = pp_sing_ex.dimension();
    let x: Vec<c64> = (0..dim).map(|i| c64::new(i as f64 * 0.01, 0.0)).collect();
    let mut y0 = vec![c64::zero(); dim];
    let mut y1 = vec![c64::zero(); dim];
    let mut y2 = vec![c64::zero(); dim];
    let mut y3 = vec![c64::zero(); dim];

    let n_iter = 100;
    
    // Warm-up
    for _ in 0..5 {
        pp_sing_ex.apply(&x, &mut y0);
        ham_sparse.apply(&x, &mut y2);
        ham_composed.apply(&x, &mut y3);
    }

    // Benchmark sing_ex.rs
    println!("Benchmarking sing_ex.rs ({} iterations)...", n_iter);
    let start = Instant::now();
    for _ in 0..n_iter {
        pp_sing_ex.apply(&x, &mut y0);
    }
    let time1 = start.elapsed();
    println!("  Average: {:?}\n", time1 / n_iter);

    // Benchmark Sparse
    println!("Benchmarking Sparse ({} iterations)...", n_iter);
    let start = Instant::now();
    for _ in 0..n_iter {
        ham_sparse.apply(&x, &mut y2);
    }
    let time2 = start.elapsed();
    println!("  Average: {:?}\n", time2 / n_iter);

    // Benchmark ComposedOperator
    println!("Benchmarking ComposedOperator ({} iterations)...", n_iter);
    let start = Instant::now();
    for _ in 0..n_iter {
        ham_composed.apply(&x, &mut y3);
    }
    let time2 = start.elapsed();
    println!("  Average: {:?}\n", time2 / n_iter);

    // Compare results
    let mut max_diff: f64 = 0.0;
    let mut max_diff_idx = 0;
    for i in 0..dim {
        let diff = (y0[i] - y2[i]).norm();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    
    // Also check diagonal elements
    println!("\nChecking diagonal elements:");
    let diag1 = pp_sing_ex.get_diagonal();
    let diag2 = ham_sparse.get_diagonal();
    let diag3 = ham_composed.get_diagonal();
    let mut max_diag_diff = 0.0f64;
    let mut max_diag_idx = 0;
    for i in 0..dim.min(10) {
        let diff = (diag1[i] - diag2[i]).norm();
        if diff > max_diag_diff {
            max_diag_diff = diff;
            max_diag_idx = i;
        }
        if i < 5 {
            println!("  diag[{}]: sing_ex={:.6}, sparse={:.6}, composed={:.6}, diff={:.2e}", 
                i, diag1[i].re, diag2[i].re, diag3[i].re, diff);
        }
    }
    println!("  Max diagonal diff: {:.6e} at index {}", max_diag_diff, max_diag_idx);
    
    // Check Hermiticity of both versions
    println!("\nChecking Hermiticity:");
    
    // // First, let's verify the KronSum structure
    // println!("  Checking Hamiltonian structure...");
    // match &ham_composed {
    //     light_matter::utils::ComposedOperator::Base(_) => println!("  Structure: Base"),
    //     light_matter::utils::ComposedOperator::Add(_, _) => println!("  Structure: Add"),
    //     light_matter::utils::ComposedOperator::Sum(_) => println!("  Structure: Sum"),
    //     light_matter::utils::ComposedOperator::Dot(_, _) => println!("  Structure: Dot"),
    //     light_matter::utils::ComposedOperator::ScalarMul(_, _) => println!("  Structure: ScalarMul"),
    //     light_matter::utils::ComposedOperator::Kron(_, _) => println!("  Structure: Kron"),
    //     light_matter::utils::ComposedOperator::Diagonal(_) => println!("  Structure: Diagonal"),
    //     light_matter::utils::ComposedOperator::Identity(_) => println!("  Structure: Identity"),
    //     light_matter::utils::ComposedOperator::Zero(_) => println!("  Structure: Zero"),
    //     light_matter::utils::ComposedOperator::Transpose(_) => println!("  Structure: Transpose"),
    //     light_matter::utils::ComposedOperator::KronSum { .. } => println!("  Structure: KronSum"),
    // }
    
    // // Check if it's a Sum containing a KronSum
    // if let light_matter::utils::ComposedOperator::Sum(terms) = &ham_composed {
    //     println!("  Hamiltonian is a Sum with {} terms", terms.len());
    //     for (i, term) in terms.iter().enumerate() {
    //         if let light_matter::utils::ComposedOperator::KronSum { transition_matrices, exciton_dim, photon_dim } = term.as_ref() {
    //             println!("  Term {}: KronSum with {} transitions ({}×{})", i, transition_matrices.len(), exciton_dim, photon_dim);
                
    //             // Check structure of this KronSum
    //             let mut unpaired = Vec::new();
    //             for (&(from, to), _) in transition_matrices.iter() {
    //                 if from != to && !transition_matrices.contains_key(&(to, from)) {
    //                     unpaired.push((from, to));
    //                 }
    //             }
    //             if !unpaired.is_empty() {
    //                 println!("    WARNING: {} unpaired transitions!", unpaired.len());
    //                 for (from, to) in unpaired.iter().take(3) {
    //                     println!("      ({}, {}) missing conjugate", from, to);
    //                 }
    //             } else {
    //                 println!("    ✓ All transitions paired");
    //             }
    //         } else {
    //             println!("  Term {}: {:?}", i, std::mem::discriminant(term.as_ref()));
    //         }
    //     }
    // } else if let light_matter::utils::ComposedOperator::KronSum { transition_matrices, exciton_dim, photon_dim } = &ham_composed {
    //     println!("  KronSum structure:");
    //     println!("    Exciton dim: {}, Photon dim: {}", exciton_dim, photon_dim);
    //     println!("    Transitions: {}", transition_matrices.len());
        
    //     // Check if transitions are paired (Hermitian structure)
    //     let mut unpaired = Vec::new();
    //     for (&(from, to), _) in transition_matrices.iter() {
    //         if from != to && !transition_matrices.contains_key(&(to, from)) {
    //             unpaired.push((from, to));
    //         }
    //     }
    //     if !unpaired.is_empty() {
    //         println!("    WARNING: {} unpaired transitions found!", unpaired.len());
    //         for (_, (from, to)) in unpaired.iter().enumerate().take(5) {
    //             println!("      ({}, {}) has no conjugate ({}, {})", from, to, to, from);
    //         }
    //     } else {
    //         println!("    ✓ All off-diagonal transitions are paired");
    //     }
        
    //     // Check if diagonal transition matrices are Hermitian
    //     let mut non_hermitian_diag = Vec::new();
    //     for (&(from, to), matrix) in transition_matrices.iter() {
    //         if from == to {
    //             let mut max_herm_err: f64 = 0.0;
    //             for i in 0..matrix.nrows() {
    //                 for j in 0..matrix.ncols() {
    //                     let diff = (matrix[[i, j]] - matrix[[j, i]].conj()).norm();
    //                     max_herm_err = max_herm_err.max(diff);
    //                 }
    //             }
    //             if max_herm_err > 1e-10 {
    //                 non_hermitian_diag.push((from, max_herm_err));
    //             }
    //         }
    //     }
    //     if !non_hermitian_diag.is_empty() {
    //         println!("    WARNING: {} non-Hermitian diagonal matrices!", non_hermitian_diag.len());
    //         for (state, err) in non_hermitian_diag.iter().take(3) {
    //             println!("      State {}: max error {:.2e}", state, err);
    //         }
    //     } else {
    //         println!("    ✓ All diagonal transition matrices are Hermitian");
    //     }
    // }

    // Check dense version
    let mut y1_transpose = vec![c64::zero(); dim];
    ham_dense.apply_transpose(&x, &mut y1_transpose);
    ham_dense.apply(&x, &mut y1);
    
    let mut herm_diff_1 = 0.0f64;
    for i in 0..dim {
        let diff = (y1[i] - y1_transpose[i].conj()).norm();
        herm_diff_1 = herm_diff_1.max(diff);
    }
    println!("  Dense: Max |H|x> - (H†|x>)*|: {:.6e}", herm_diff_1);
    println!("  Dense:  H|x> = {:?}", &y1[0..3]);
    println!("  Dense: HT|x> = {:?}", &y1_transpose[0..3]);
    
    // Check sparse version
    let mut y2_transpose = vec![c64::zero(); dim];
    ham_sparse.apply_transpose(&x, &mut y2_transpose);
    
    let mut herm_diff_2 = 0.0f64;
    for i in 0..dim {
        let diff = (y2[i] - y2_transpose[i].conj()).norm();
        herm_diff_2 = herm_diff_2.max(diff);
    }
    println!("  Sparse: Max |H|x> - (H†|x>)*|: {:.6e}", herm_diff_2);
    println!("  Sparse:  H|x> = {:?}", &y2[0..3]);
    println!("  Sparse: HT|x> = {:?}", &y2_transpose[0..3]);
    
    // Check ComposedOperator version
    let mut y3_transpose = vec![c64::zero(); dim];
    ham_composed.apply_transpose(&x, &mut y3_transpose);
    
    let mut herm_diff_3 = 0.0f64;
    for i in 0..dim {
        let diff = (y3[i] - y3_transpose[i].conj()).norm();
        herm_diff_3 = herm_diff_3.max(diff);
    }
    println!("  ComposedOp: Max |H|x> - (H†|x>)*|: {:.6e}", herm_diff_3);
    println!("  ComposedOp:  H|x> = {:?}", &y3[0..3]);
    println!("  ComposedOp: HT|x> = {:?}", &y3_transpose[0..3]);
    
    if herm_diff_1 < 1e-10 && herm_diff_2 < 1e-10 && herm_diff_3 < 1e-10 {
        println!("  ✓ All Hamiltonians are Hermitian");
    } 
    if herm_diff_1 > 1e-10 {
        println!("  ✗ WARNING: Dense is not Hermitian!");
    } 
    if herm_diff_2 > 1e-10 {
        println!("  ✗ WARNING: Sparse is not Hermitian!");
    } 
    if herm_diff_3 > 1e-10 {
        println!("  ✗ WARNING: ComposedOp is not Hermitian!");
    }

    println!("Results:");
    println!("  Maximum difference: {:.2e} at index {}", max_diff, max_diff_idx);
    println!("    sing_ex: {:?}", y0[max_diff_idx]);
    println!("    Sparse: {:?}", y2[max_diff_idx]);
    
    // Show a few more entries around the max diff
    println!("  Context around max diff:");
    for i in max_diff_idx.saturating_sub(2)..=(max_diff_idx + 2).min(dim - 1) {
        println!("    [{}] sing_ex: {:?}, Sparse: {:?}, diff: {:.2e}", 
            i, y0[i], y2[i], (y0[i] - y2[i]).norm());
    }
    
    let speedup = time1.as_secs_f64() / time2.as_secs_f64();
    if speedup > 1.0 {
        println!("  ComposedOperator is {:.2}× FASTER", speedup);
    } else {
        println!("  ComposedOperator is {:.2}× slower", 1.0 / speedup);
    }

    if max_diff < 1e-6 {
        println!("  ✓ Results match!\n");
    } else {
        println!("  ✗ Results differ!\n");
    }

    println!("Benchmark complete!");
}
