/// Minimal Hamiltonian tests focusing on Bosonic systems
///
/// This test suite verifies basic Hamiltonian construction without
/// requiring complex Exciton configurations

use ndarray::{Array2, arr1}; 
use light_matter::utils::{
    BosonMode, Vector3D, Polarization,
};
use light_matter::hamiltonians::{BosonicHamiltonian, FullBosons, SingExBosons, ExcitonicHamiltonian};
use exciton_2d::{ExcitonConfig, Exciton2DGridSystem};
use std::path::PathBuf;

const EPSILON: f64 = 1e-8;

// ============================================================================
// BosonMode Tests
// ============================================================================

#[test]
fn test_boson_mode_creation() {
    let q = Vector3D::new(0.01, 0.0, 0.0);
    let mode = BosonMode::new(q, Polarization::TE, 3);
    
    // Frequency should be |q| * c
    let expected_omega = q.magnitude() * 137.036; // c in atomic units
    assert!((mode.omega - expected_omega).abs() < EPSILON);
    
    // Wavevector should be projected to 2D
    assert!((mode.q[0] - 0.01).abs() < EPSILON);
    assert!(mode.q[1].abs() < EPSILON);
    
    // Polarization should be normalized
    assert!((mode.eps.iter().map(|x| x * x).sum::<f64>().sqrt() - 1.0).abs() < EPSILON);
    
    // Fock space dimension
    assert_eq!(mode.n_fock, 3);
}

#[test]
fn test_boson_mode_te_polarization() {
    let q = Vector3D::new(0.01, 0.0, 1.0);
    let mode_te = BosonMode::new(q, Polarization::TE, 2);
    
    // For q_y = 0, TE should be along y
    assert!(mode_te.eps[0].abs() < EPSILON);
    assert!((mode_te.eps[1].abs() - 1.0).abs() < EPSILON);
}

#[test]
fn test_boson_mode_tm_polarization() {
    let q = Vector3D::new(0.01, 0.0, 1.0);
    let mode_tm = BosonMode::new(q, Polarization::TM, 2);
    
    // For q_x > 0, TM should have no y component
    assert!(mode_tm.eps[1].abs() < EPSILON);
}

#[test]
fn test_vector3d_operations() {
    let v1 = Vector3D::new(1.0, 0.0, 0.0);
    let v2 = Vector3D::new(0.0, 1.0, 0.0);
    
    // Test magnitude
    assert!((v1.magnitude() - 1.0).abs() < EPSILON);
    
    // Test cross product
    let cross = v1.cross(&v2);
    assert!(cross.x.abs() < EPSILON);
    assert!(cross.y.abs() < EPSILON);
    assert!((cross.z - 1.0).abs() < EPSILON);
    
    // Test normalization
    let v3 = Vector3D::new(3.0, 4.0, 0.0);
    let normalized = v3.normalize();
    assert!((normalized.magnitude() - 1.0).abs() < EPSILON);
    assert!((normalized.x - 0.6).abs() < EPSILON);
    assert!((normalized.y - 0.8).abs() < EPSILON);
}

// ============================================================================
// Full Bosonic Hamiltonian Tests
// ============================================================================

#[test]
fn test_full_bosons_single_mode() {
    let q = Vector3D::new(0.01, 0.0, 0.0);
    let mode = BosonMode::new(q, Polarization::TE, 3);
    let omega = mode.omega;
    
    let bosonic_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
        BosonicHamiltonian::new(vec![mode]);
    
    // Dimension should be n_fock
    assert_eq!(bosonic_ham.dimension(), 3);
    
    // Get Hamiltonian
    let h = bosonic_ham.get_h();
    
    // Should be diagonal with eigenvalues 0, ω, 2ω
    assert!((h[[0, 0]] - 0.0).abs() < EPSILON);
    assert!((h[[1, 1]] - omega).abs() < EPSILON);
    assert!((h[[2, 2]] - 2.0 * omega).abs() < EPSILON);
    
    // Off-diagonal should be zero
    assert!(h[[0, 1]].abs() < EPSILON);
    assert!(h[[1, 0]].abs() < EPSILON);
}

#[test]
fn test_full_bosons_annihilation_operator() {
    let q = Vector3D::new(0.01, 0.0, 0.0);
    let mode = BosonMode::new(q, Polarization::TE, 4);
    
    let bosonic_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
        BosonicHamiltonian::new(vec![mode]);
    
    let a = bosonic_ham.get_a(0);
    
    // Check dimensions
    assert_eq!(a.shape(), &[4, 4]);
    
    // a|n⟩ = √n |n-1⟩
    // So a should have √n on the subdiagonal
    assert!(a[[0, 0]].abs() < EPSILON);
    assert!((a[[0, 1]] - 1.0_f64.sqrt()).abs() < EPSILON);
    assert!((a[[1, 2]] - 2.0_f64.sqrt()).abs() < EPSILON);
    assert!((a[[2, 3]] - 3.0_f64.sqrt()).abs() < EPSILON);
    
    // All other elements should be zero
    assert!(a[[1, 0]].abs() < EPSILON);
    assert!(a[[2, 0]].abs() < EPSILON);
}

#[test]
fn test_full_bosons_number_operator() {
    let q = Vector3D::new(0.01, 0.0, 0.0);
    let mode = BosonMode::new(q, Polarization::TE, 4);
    
    let bosonic_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
        BosonicHamiltonian::new(vec![mode]);
    
    let n_op = bosonic_ham.get_adag_a(0);
    
    // Should be diagonal with eigenvalues 0, 1, 2, 3
    for i in 0..4 {
        assert!((n_op[[i, i]] - i as f64).abs() < EPSILON);
        
        // Off-diagonal should be zero
        for j in 0..4 {
            if i != j {
                assert!(n_op[[i, j]].abs() < EPSILON);
            }
        }
    }
}

#[test]
fn test_full_bosons_two_modes() {
    let q1 = Vector3D::new(0.01, 0.0, 0.0);
    let q2 = Vector3D::new(0.0, 0.01, 0.0);
    let mode1 = BosonMode::new(q1, Polarization::TE, 2);
    let mode2 = BosonMode::new(q2, Polarization::TE, 3);
    
    let bosonic_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
        BosonicHamiltonian::new(vec![mode1, mode2]);
    
    // Dimension should be n_fock1 * n_fock2
    assert_eq!(bosonic_ham.dimension(), 6);
    
    // Get Hamiltonian
    let h = bosonic_ham.get_h();
    
    // Should be 6x6
    assert_eq!(h.shape(), &[6, 6]);
    
    // Check that it's Hermitian (symmetric for real matrices)
    for i in 0..6 {
        for j in 0..6 {
            assert!((h[[i, j]] - h[[j, i]]).abs() < EPSILON);
        }
    }
}

#[test]
fn test_hamiltonian_hermiticity() {
    // Test that bosonic Hamiltonians are Hermitian
    let q = Vector3D::new(0.01, 0.01, 0.0);
    let mode = BosonMode::new(q, Polarization::TE, 3);
    let bosonic_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
        BosonicHamiltonian::new(vec![mode]);
    
    let h = bosonic_ham.get_h();
    
    // Check Hermiticity: H[i,j] = H[j,i]
    for i in 0..h.nrows() {
        for j in 0..h.ncols() {
            let diff = (h[[i, j]] - h[[j, i]]).abs();
            assert!(
                diff < EPSILON,
                "Hamiltonian not Hermitian at ({}, {}): {} vs {} (diff: {})",
                i, j, h[[i, j]], h[[j, i]], diff
            );
        }
    }
}

// ============================================================================
// Sparse vs Dense Consistency Tests
// ============================================================================

#[test]
fn test_sparse_dense_hamiltonian_equivalence() {
    use sprs::CsMat;
    use light_matter::hamiltonians::FullBosons as FullBosonsSparse;
    
    let q = Vector3D::new(0.01, 0.0, 0.0);
    let mode = BosonMode::new(q, Polarization::TE, 3);
    
    // Build with dense matrices
    let dense_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
        BosonicHamiltonian::new(vec![mode.clone()]);
    let h_dense = dense_ham.get_h();
    
    // Build with sparse matrices
    let sparse_ham: BosonicHamiltonian<CsMat<f64>, FullBosonsSparse<CsMat<f64>>> = 
        BosonicHamiltonian::new(vec![mode]);
    let h_sparse = sparse_ham.get_h();
    
    // Convert sparse to dense for comparison
    let h_sparse_dense = h_sparse.to_dense();
    
    // Compare
    for i in 0..h_dense.nrows() {
        for j in 0..h_dense.ncols() {
            assert!(
                (h_dense[[i, j]] - h_sparse_dense[[i, j]]).abs() < EPSILON,
                "Dense and sparse differ at ({}, {}): {} vs {}",
                i, j, h_dense[[i, j]], h_sparse_dense[[i, j]]
            );
        }
    }
}


// ============================================================================
// Single Excitation Bosonic Hamiltonian Tests
// ============================================================================

#[test]
fn test_sing_ex_bosons_dimension() {
    let q = Vector3D::new(0.01, 0.0, 0.0);
    let mode1 = BosonMode::new(q, Polarization::TE, 2);
    let mode2 = BosonMode::new(q, Polarization::TM, 2);
    
    let bosonic_ham: BosonicHamiltonian<Array2<f64>, SingExBosons<Array2<f64>>> = 
        BosonicHamiltonian::new(vec![mode1, mode2]);
    
    // Dimension should be n_modes + 1 (ground state + single excitations)
    assert_eq!(bosonic_ham.dimension(), 3);
}

#[test]
fn test_sing_ex_bosons_annihilation() {
    let q = Vector3D::new(0.01, 0.0, 0.0);
    let mode1 = BosonMode::new(q, Polarization::TE, 2);
    let mode2 = BosonMode::new(q, Polarization::TM, 2);
    
    let bosonic_ham: BosonicHamiltonian<Array2<f64>, SingExBosons<Array2<f64>>> = 
        BosonicHamiltonian::new(vec![mode1, mode2]);
    
    let a0 = bosonic_ham.get_a(0);
    let a1 = bosonic_ham.get_a(1);
    
    // Both should be 3x3
    assert_eq!(a0.shape(), &[3, 3]);
    assert_eq!(a1.shape(), &[3, 3]);
    
    // a_i |0⟩ = 0, a_i |j⟩ = δ_ij |0⟩
    // So a_i should annihilate ground state and map |i⟩ to |0⟩
    
    // Check a0
    assert!(a0[[0, 0]].abs() < EPSILON); // ⟨0|a0|0⟩ = 0
    assert!((a0[[0, 1]] - 1.0).abs() < EPSILON); // ⟨0|a0|1⟩ = 1
    assert!(a0[[0, 2]].abs() < EPSILON); // ⟨0|a0|2⟩ = 0
    
    // Check a1
    assert!(a1[[0, 0]].abs() < EPSILON);
    assert!(a1[[0, 1]].abs() < EPSILON);
    assert!((a1[[0, 2]] - 1.0).abs() < EPSILON);
}

#[test]
fn test_sing_ex_hamiltonian() {
    let q = Vector3D::new(0.01, 0.0, 0.0);
    let mode = BosonMode::new(q, Polarization::TE, 2);
    let omega = mode.omega;
    
    let bosonic_ham: BosonicHamiltonian<Array2<f64>, SingExBosons<Array2<f64>>> = 
        BosonicHamiltonian::new(vec![mode]);
    
    let h = bosonic_ham.get_h();
    
    // Should be 2x2: |0⟩ and |1⟩
    assert_eq!(h.shape(), &[2, 2]);
    
    // H = ω a†a, so eigenvalues are 0 and ω
    assert!(h[[0, 0]].abs() < EPSILON);
    assert!((h[[1, 1]] - omega).abs() < EPSILON);
}

// ============================================================================
// Excitonic Hamiltonian Tests
// ============================================================================

#[test]
fn test_excitonic_hamiltonian_initialization() {
    // Create a minimal exciton configuration
    let config = ExcitonConfig {
        n_kappa: 7,                               // Number of bands
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
        n_jobs: 32,                               // Number of parallel jobs
        use_gpu: false,                           // Whether to use GPU acceleration
        batch_size: 32,                           // Batch size for calculations
    };
    
    let exciton_grid = Exciton2DGridSystem::from_config(config);
    let mut exciton_ham: ExcitonicHamiltonian<Array2<f64>> = 
        ExcitonicHamiltonian::new(exciton_grid.full_ham, 3, 3);
    
    // Should start empty
    assert!(exciton_ham.is_empty());
    
    // After computing eigenstates, should not be empty
    exciton_ham.compute_eigenstates();
    assert!(!exciton_ham.is_empty());
    
    // Should have computed the requested number of eigenstates
    assert_eq!(exciton_ham.dimension(), 3);
}

#[test]
fn test_excitonic_hamiltonian_properties() {
    let config = ExcitonConfig {
        n_kappa: 7,                               // Number of bands
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
        n_jobs: 32,                               // Number of parallel jobs
        use_gpu: false,                           // Whether to use GPU acceleration
        batch_size: 32,                           // Batch size for calculations
    };
    
    let exciton_grid = Exciton2DGridSystem::from_config(config);
    let mut exciton_ham: ExcitonicHamiltonian<Array2<f64>> = 
        ExcitonicHamiltonian::new(exciton_grid.full_ham, 3, 3);
    
    exciton_ham.compute_eigenstates();
    
    // Get Hamiltonian
    let h = exciton_ham.get_h();
    
    // Should be diagonal in eigenstate basis
    assert_eq!(h.shape(), &[3, 3]);
    
    // Check it's diagonal
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                assert!(h[[i, j]].abs() < EPSILON);
            }
        }
    }
    
    // Eigenvalues should be in ascending order (ground state first)
    let e_vals = exciton_ham.e_vals.as_ref().unwrap();
    for i in 0..e_vals.len()-1 {
        assert!(e_vals[i] <= e_vals[i+1], "Eigenvalues not sorted: {} > {}", e_vals[i], e_vals[i+1]);
    }
}

#[test]
fn test_excitonic_momentum_operator() {
    let config = ExcitonConfig {
        n_kappa: 7,                               // Number of bands
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
        n_jobs: 32,                               // Number of parallel jobs
        use_gpu: false,                           // Whether to use GPU acceleration
        batch_size: 32,                           // Batch size for calculations
    };
    
    let exciton_grid = Exciton2DGridSystem::from_config(config);
    let mut exciton_ham: ExcitonicHamiltonian<Array2<f64>> = 
        ExcitonicHamiltonian::new(exciton_grid.full_ham, 3, 3);
    
    exciton_ham.compute_eigenstates();
    
    // Get momentum operator in x direction
    let field_x = arr1(&[1.0, 0.0]);
    let p_x = exciton_ham.get_p_com(&field_x, false);
    
    // Should be 3x3
    assert_eq!(p_x.shape(), &[3, 3]);
    
    // For real symmetric systems, momentum matrix elements should be real
    for i in 0..3 {
        for j in 0..3 {
            // Check that matrix is Hermitian
            assert!((p_x[[i, j]] - p_x[[j, i]]).abs() < EPSILON);
        }
    }
}

// // ============================================================================
// // Light-Matter Hamiltonian Tests
// // ============================================================================

// #[test]
// fn test_lm_hamiltonian_uncoupled_limit() {
//     // Test that with zero coupling, we get independent exciton and photon
//     let config = ExcitonConfig {
//         n_kappa_x: 3,
//         n_kappa_y: 3,
//         v: 0.05,
//         a: 0.9,
//         delta: 0.0,
//         m: 1.0,
//         mu: 0.05,
//     };
    
//     let exciton_grid = Exciton2DGridSystem::from_config(config);
//     let exciton_ham: ExcitonicHamiltonian<Array2<f64>> = 
//         ExcitonicHamiltonian::new(exciton_grid.full_ham, 2, 3);
    
//     let q = Vector3D::new(0.01, 0.0, 0.0);
//     let mode = BosonMode::new(q, Polarization::TE, 2);
//     let photon_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
//         BosonicHamiltonian::new(vec![mode]);
    
//     // Zero coupling
//     let mut lm_ham = LMHamiltonian::new(exciton_ham, photon_ham, 0.0);
    
//     let h = lm_ham.build_lm_hamiltonian(None);
    
//     // Should be (n_ex * n_ph) x (n_ex * n_ph) = 4x4
//     assert_eq!(h.shape(), (4, 4));
    
//     // With zero coupling, Hamiltonian should be block diagonal
//     // Check that off-diagonal blocks are small
//     let mut max_off_diagonal = 0.0;
//     for i in 0..4 {
//         for j in 0..4 {
//             if (i / 2) != (j / 2) { // Different blocks
//                 max_off_diagonal = max_off_diagonal.max(h[[i, j]].abs());
//             }
//         }
//     }
    
//     // In the uncoupled case, there will still be A^2 terms
//     // So we can't expect perfect block diagonal structure
//     // Just check that the coupling is weak
//     println!("Max off-diagonal element with zero coupling: {}", max_off_diagonal);
// }

// #[test]
// fn test_lm_hamiltonian_hermiticity() {
//     let config = ExcitonConfig {
//         n_kappa_x: 3,
//         n_kappa_y: 3,
//         v: 0.05,
//         a: 0.9,
//         delta: 0.0,
//         m: 1.0,
//         mu: 0.05,
//     };
    
//     let exciton_grid = Exciton2DGridSystem::from_config(config);
//     let exciton_ham: ExcitonicHamiltonian<Array2<f64>> = 
//         ExcitonicHamiltonian::new(exciton_grid.full_ham, 2, 3);
    
//     let q = Vector3D::new(0.01, 0.0, 0.0);
//     let mode = BosonMode::new(q, Polarization::TE, 2);
//     let photon_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
//         BosonicHamiltonian::new(vec![mode]);
    
//     let mut lm_ham = LMHamiltonian::new(exciton_ham, photon_ham, 0.01);
    
//     let h = lm_ham.build_lm_hamiltonian(None);
    
//     // Check Hermiticity
//     for i in 0..h.nrows() {
//         for j in 0..h.ncols() {
//             let diff = (h[[i, j]] - h[[j, i]]).abs();
//             assert!(
//                 diff < EPSILON,
//                 "Hamiltonian not Hermitian at ({}, {}): {} vs {}",
//                 i, j, h[[i, j]], h[[j, i]]
//             );
//         }
//     }
// }

// #[test]
// fn test_lm_hamiltonian_k_dependence() {
//     let config = ExcitonConfig {
//         n_kappa_x: 3,
//         n_kappa_y: 3,
//         v: 0.05,
//         a: 0.9,
//         delta: 0.0,
//         m: 1.0,
//         mu: 0.05,
//     };
    
//     let exciton_grid = Exciton2DGridSystem::from_config(config);
//     let exciton_ham: ExcitonicHamiltonian<Array2<f64>> = 
//         ExcitonicHamiltonian::new(exciton_grid.full_ham, 2, 3);
    
//     let q = Vector3D::new(0.01, 0.0, 0.0);
//     let mode = BosonMode::new(q, Polarization::TE, 2);
//     let photon_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
//         BosonicHamiltonian::new(vec![mode]);
    
//     let mut lm_ham = LMHamiltonian::new(exciton_ham, photon_ham, 0.01);
    
//     // Build at k = 0
//     let h1 = lm_ham.build_lm_hamiltonian(Some(vec![0.0, 0.0]));
    
//     // Build at different k
//     let h2 = lm_ham.build_lm_hamiltonian(Some(vec![0.01, 0.0]));
    
//     // Hamiltonians should be different
//     let mut differ = false;
//     for i in 0..h1.nrows() {
//         for j in 0..h1.ncols() {
//             if (h1[[i, j]] - h2[[i, j]]).abs() > EPSILON {
//                 differ = true;
//                 break;
//             }
//         }
//         if differ {
//             break;
//         }
//     }
    
//     assert!(differ, "Hamiltonian should depend on k");
// }

// // ============================================================================
// // Polaron-Polariton Hamiltonian Tests
// // ============================================================================

// #[test]
// fn test_pp_hamiltonian_from_config() {
//     let e_config = ExcitonConfig {
//         n_kappa_x: 3,
//         n_kappa_y: 3,
//         v: 0.05,
//         a: 0.9,
//         delta: 0.0,
//         m: 1.0,
//         mu: 0.05,
//     };
    
//     let pp_config = PPConfig::new(
//         e_config,
//         3,  // n_rel
//         2,  // n_ex_eig
//         0.01,  // lm_coupling
//         0.0,   // omega_0 (not used here)
//         vec![Vector3D::new(0.01, 0.0, 0.0)],  // q_values
//         0.001,  // pn_coupling
//         0.02,   // pn_omega
//         arr1(&[0.02, 0.0]),  // pn_k
//         2,  // pn_n_fock
//         10,  // n_k_points
//         5,   // n_bands
//     );
    
//     let mut pp_ham: PPHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
//         PPHamiltonian::from_config(&pp_config);
    
//     // Check dimensions
//     let n_ex = 2;
//     let n_ph = 2;  // 1 photon mode with n_fock = 2
//     let n_pn = 2;  // phonon with n_fock = 2
    
//     let h = pp_ham.build_pp_hamiltonian(None);
    
//     let expected_dim = n_ex * n_ph * n_pn;
//     assert_eq!(h.shape(), (expected_dim, expected_dim));
// }

// #[test]
// fn test_pp_hamiltonian_hermiticity() {
//     let e_config = ExcitonConfig {
//         n_kappa_x: 3,
//         n_kappa_y: 3,
//         v: 0.05,
//         a: 0.9,
//         delta: 0.0,
//         m: 1.0,
//         mu: 0.05,
//     };
    
//     let pp_config = PPConfig::new(
//         e_config,
//         3,
//         2,
//         0.01,
//         0.0,
//         vec![Vector3D::new(0.01, 0.0, 0.0)],
//         0.001,
//         0.02,
//         arr1(&[0.02, 0.0]),
//         2,
//         10,
//         5,
//     );
    
//     let mut pp_ham: PPHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
//         PPHamiltonian::from_config(&pp_config);
    
//     let h = pp_ham.build_pp_hamiltonian(None);
    
//     // Check Hermiticity
//     for i in 0..h.nrows() {
//         for j in 0..h.ncols() {
//             let diff = (h[[i, j]] - h[[j, i]]).abs();
//             assert!(
//                 diff < EPSILON,
//                 "PP Hamiltonian not Hermitian at ({}, {}): {} vs {} (diff: {})",
//                 i, j, h[[i, j]], h[[j, i]], diff
//             );
//         }
//     }
// }

// #[test]
// fn test_pp_hamiltonian_update_k() {
//     let e_config = ExcitonConfig {
//         n_kappa_x: 3,
//         n_kappa_y: 3,
//         v: 0.05,
//         a: 0.9,
//         delta: 0.0,
//         m: 1.0,
//         mu: 0.05,
//     };
    
//     let pp_config = PPConfig::new(
//         e_config,
//         3,
//         2,
//         0.01,
//         0.0,
//         vec![Vector3D::new(0.01, 0.0, 0.0)],
//         0.001,
//         0.02,
//         arr1(&[0.02, 0.0]),
//         2,
//         10,
//         5,
//     );
    
//     let mut pp_ham: PPHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
//         PPHamiltonian::from_config(&pp_config);
    
//     let h1 = pp_ham.build_pp_hamiltonian(Some(vec![0.0, 0.0]));
    
//     pp_ham.update_k(vec![0.01, 0.0]);
//     let h2 = pp_ham.build_pp_hamiltonian(None);
    
//     // Should produce different Hamiltonians
//     let mut differ = false;
//     for i in 0..h1.nrows().min(5) {  // Check first few elements
//         for j in 0..h1.ncols().min(5) {
//             if (h1[[i, j]] - h2[[i, j]]).abs() > EPSILON {
//                 differ = true;
//                 break;
//             }
//         }
//         if differ {
//             break;
//         }
//     }
    
//     assert!(differ, "PP Hamiltonian should change with k");
// }

// // ============================================================================
// // Integration Tests
// // ============================================================================

// #[test]
// fn test_energy_conservation() {
//     // Test that total energy is conserved in the uncoupled limit
//     let config = ExcitonConfig {
//         n_kappa_x: 3,
//         n_kappa_y: 3,
//         v: 0.05,
//         a: 0.9,
//         delta: 0.0,
//         m: 1.0,
//         mu: 0.05,
//     };
    
//     let exciton_grid = Exciton2DGridSystem::from_config(config);
//     let mut exciton_ham: ExcitonicHamiltonian<Array2<f64>> = 
//         ExcitonicHamiltonian::new(exciton_grid.full_ham, 2, 3);
    
//     exciton_ham.compute_eigenstates();
//     let ex_h = exciton_ham.get_h();
    
//     let q = Vector3D::new(0.01, 0.0, 0.0);
//     let mode = BosonMode::new(q, Polarization::TE, 2);
//     let photon_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
//         BosonicHamiltonian::new(vec![mode]);
    
//     let ph_h = photon_ham.get_h();
    
//     // In uncoupled limit, total energy should be sum of exciton and photon energies
//     // H_total = H_ex ⊗ I_ph + I_ex ⊗ H_ph
    
//     let ex_eye = Array2::eye(2);
//     let ph_eye = Array2::eye(2);
    
//     let h_ex_part = ex_h.kron(&ph_eye);
//     let h_ph_part = ex_eye.kron(&ph_h);
//     let h_uncoupled = h_ex_part.add(&h_ph_part);
    
//     // Check that trace is conserved (sum of diagonal elements)
//     let mut trace = 0.0;
//     for i in 0..h_uncoupled.nrows() {
//         trace += h_uncoupled[[i, i]];
//     }
    
//     // Trace should be sum of individual traces
//     let mut ex_trace = 0.0;
//     for i in 0..ex_h.nrows() {
//         ex_trace += ex_h[[i, i]];
//     }
    
//     let mut ph_trace = 0.0;
//     for i in 0..ph_h.nrows() {
//         ph_trace += ph_h[[i, i]];
//     }
    
//     let expected_trace = ex_trace * ph_h.nrows() as f64 + ph_trace * ex_h.nrows() as f64;
    
//     assert!(
//         (trace - expected_trace).abs() < EPSILON,
//         "Trace not conserved: {} vs {}",
//         trace,
//         expected_trace
//     );
// }

// #[test]
// fn test_dimension_consistency() {
//     // Test that dimensions are consistent across different representations
//     let e_config = ExcitonConfig {
//         n_kappa_x: 3,
//         n_kappa_y: 3,
//         v: 0.05,
//         a: 0.9,
//         delta: 0.0,
//         m: 1.0,
//         mu: 0.05,
//     };
    
//     let pp_config = PPConfig::new(
//         e_config,
//         3,
//         2,
//         0.01,
//         0.0,
//         vec![Vector3D::new(0.01, 0.0, 0.0)],
//         0.001,
//         0.02,
//         arr1(&[0.02, 0.0]),
//         2,
//         10,
//         5,
//     );
    
//     let mut pp_ham: PPHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
//         PPHamiltonian::from_config(&pp_config);
    
//     let n_ex = pp_ham.polariton.exciton.dimension();
//     let n_ph = pp_ham.polariton.photon.dimension();
//     let n_pn = pp_ham.phonon.dimension();
    
//     let h = pp_ham.build_pp_hamiltonian(None);
    
//     assert_eq!(h.shape().0, n_ex * n_ph * n_pn);
//     assert_eq!(h.shape().1, n_ex * n_ph * n_pn);
// }
