/// Tests for LMMatrixElementArray Hermiticity properties
///
/// This test suite verifies that the matrix elements computed for
/// light-matter coupling satisfy the correct Hermiticity properties:
/// - cos_mat and sin_mat should be Hermitian (H† = H)
/// - p_cos_mat should be anti-Hermitian (H† = -H)

use ndarray::Array2;
use ndarray_linalg::c64;
use light_matter::utils::{
    BosonMode, Polarization, PrettyC64Array2, Vector3D
};
use light_matter::hamiltonians::{
    BosonicHamiltonian, FullBosons, ExcitonicHamiltonian, LMHamiltonian,
};
use exciton_2d::{ExcitonConfig, Exciton2DGridSystem};
use std::path::PathBuf;

const EPSILON: f64 = 1e-8;

/// Helper function to check if a complex matrix is Hermitian
fn is_hermitian(matrix: &Array2<c64>, tolerance: f64) -> bool {
    let (nrows, ncols) = matrix.dim();
    if nrows != ncols {
        return false;
    }
    
    for i in 0..nrows {
        for j in 0..ncols {
            let diff = (matrix[[i, j]] - matrix[[j, i]].conj()).norm();
            if diff > tolerance {
                println!("Hermitian violation at ({}, {}): {} vs {} (diff: {})", 
                    i, j, matrix[[i, j]], matrix[[j, i]].conj(), diff);
                return false;
            }
        }
    }
    true
}

/// Helper function to check if a complex matrix is anti-Hermitian
fn is_anti_hermitian(matrix: &Array2<c64>, tolerance: f64) -> bool {
    let (nrows, ncols) = matrix.dim();
    if nrows != ncols {
        return false;
    }
    
    println!("Antihermitian test for: \n{}", PrettyC64Array2(matrix,2));

    for i in 0..nrows {
        for j in 0..ncols {
            let diff = (matrix[[i, j]] + matrix[[j, i]].conj()).norm();
            if diff > tolerance {
                println!("Anti-Hermitian violation at ({}, {}): {} vs {} (diff: {})", 
                    i, j, matrix[[i, j]], -matrix[[j, i]], diff);
                return false;
            }
        }
    }
    true
}

#[test]
fn test_matrix_elements_hermiticity_single_mode() {
    // Create a minimal exciton configuration
    let config = ExcitonConfig {
        n_kappa: 7,
        a0: 9.0,
        v0: 0.05,
        k: vec![0.0, 0.0],
        m_e: 0.3,
        m_h: 0.3,
        verbose: false,
        grid_size: 64,
        box_size: 3.0,
        save_dir: PathBuf::from("saved_data_lm"),
        recalculate: false,
        n_jobs: 32,
        use_gpu: false,
        batch_size: 32,
    };
    
    let exciton_grid = Exciton2DGridSystem::from_config(config);
    let mu = exciton_grid.rel_ham.mu;
    let exciton_ham: ExcitonicHamiltonian<Array2<f64>> = 
        ExcitonicHamiltonian::new(exciton_grid.full_ham, 3, 9);
    
    // Create a single photon mode
    let q = Vector3D::new(0.01, 0.04, 0.0);
    let mode = BosonMode::new(q, Polarization::TE, 2);
    let photon_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
        BosonicHamiltonian::new(vec![mode.clone()]);
    
    // Create LM Hamiltonian with small coupling
    let mut lm_ham = LMHamiltonian::new(exciton_ham, photon_ham, 0.01);
    
    // Initialize to compute matrix elements
    lm_ham.initialize(None);
    
    // Get the matrix elements
    let matrix_elements = &lm_ham.matrix_elements;

    let (_,sin_mat,p_cos_mat) = lm_ham.matrix_elements.compute_relative_tay_mat_elements(&vec![vec![0]], &vec![mode], 9, mu);
    
    // Check that matrix elements were computed
    assert!(!matrix_elements.is_empty(), "Matrix elements should be computed");
    
    // Test cos_mat Hermiticity
    println!("\nTesting cos_mat Hermiticity:");
    for (key, matrix) in matrix_elements.cos_mat.iter() {
        println!("  Checking cos_mat for key {:?}", key);
        assert!(
            is_hermitian(matrix, EPSILON),
            "cos_mat for key {:?} is not Hermitian",
            key
        );
    }
    
    // Test sin_mat Hermiticity
    println!("\nTesting sin_mat Hermiticity:");
    for (key, matrix) in sin_mat.iter() {
        println!("  Checking sin_mat for key {:?}", key);
        assert!(
            is_hermitian(matrix, EPSILON),
            "sin_mat for key {:?} is not Hermitian",
            key
        );
    }
    
    // Test p_cos_mat anti-Hermiticity
    println!("\nTesting p_cos_mat anti-Hermiticity:");
    for (key, matrix) in p_cos_mat.iter() {
        // println!("  Checking p_cos_mat for key {:?}", key);
        // assert!(
        //     is_anti_hermitian(&(matrix), EPSILON),
        //     "p_cos_mat for key {:?} is not anti-Hermitian",
        //     key
        // );

        let real_basis = matrix_elements.to_real_rel_basis(&matrix);
        let exciton = matrix_elements.to_exciton_basis(&real_basis, &lm_ham.exciton.e_vecs.as_ref().unwrap(), lm_ham.exciton.exciton_ham.kappa_vectors.len());

        println!("  Checking p_cos_mat in real basis for key {:?}", key);
        assert!(
            is_anti_hermitian(&(exciton), EPSILON),
            "p_cos_mat for key {:?} is not anti-Hermitian",
            key
        );
    }

    // panic!("Placeholder to see test output");
}

#[test]
fn test_matrix_elements_hermiticity_multiple_modes() {
    // Create a minimal exciton configuration
    let config = ExcitonConfig {
        n_kappa: 7,
        a0: 9.0,
        v0: 0.05,
        k: vec![0.0, 0.0],
        m_e: 0.3,
        m_h: 0.3,
        verbose: false,
        grid_size: 64,
        box_size: 3.0,
        save_dir: PathBuf::from("saved_data_lm"),
        recalculate: false,
        n_jobs: 32,
        use_gpu: false,
        batch_size: 32,
    };
    
    let exciton_grid = Exciton2DGridSystem::from_config(config);
    let exciton_ham: ExcitonicHamiltonian<Array2<f64>> = 
        ExcitonicHamiltonian::new(exciton_grid.full_ham, 3, 3);
    
    // Create multiple photon modes
    let q1 = Vector3D::new(0.01, 0.0, 0.0);
    let q2 = Vector3D::new(0.0, 0.01, 0.0);
    let mode1 = BosonMode::new(q1, Polarization::TE, 2);
    let mode2 = BosonMode::new(q2, Polarization::TE, 2);
    let photon_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
        BosonicHamiltonian::new(vec![mode1, mode2]);
    
    // Create LM Hamiltonian
    let mut lm_ham = LMHamiltonian::new(exciton_ham, photon_ham, 0.01);
    
    // Initialize to compute matrix elements
    lm_ham.initialize(None);
    
    // Get the matrix elements
    let matrix_elements = &lm_ham.matrix_elements;
    
    // Check that matrix elements were computed
    assert!(!matrix_elements.is_empty(), "Matrix elements should be computed");
    
    // Test cos_mat Hermiticity (including A² terms)
    println!("\nTesting cos_mat Hermiticity with multiple modes:");
    for (key, matrix) in matrix_elements.cos_mat.iter() {
        println!("  Checking cos_mat for key {:?}", key);
        assert!(
            is_hermitian(matrix, EPSILON),
            "cos_mat for key {:?} is not Hermitian",
            key
        );
    }
    
    // Test sin_mat Hermiticity
    println!("\nTesting sin_mat Hermiticity with multiple modes:");
    for (key, matrix) in matrix_elements.sin_mat.iter() {
        println!("  Checking sin_mat for key {:?}", key);
        assert!(
            is_hermitian(matrix, EPSILON),
            "sin_mat for key {:?} is not Hermitian",
            key
        );
    }
    
    // Test p_cos_mat anti-Hermiticity
    println!("\nTesting p_cos_mat anti-Hermiticity with multiple modes:");
    for (key, matrix) in matrix_elements.p_cos_mat.iter() {
        println!("  Checking p_cos_mat for key {:?}", key);
        assert!(
            is_anti_hermitian(matrix, EPSILON),
            "p_cos_mat for key {:?} is not anti-Hermitian",
            key
        );
    }
}

#[test]
fn test_matrix_elements_hermiticity_different_k() {
    // Test that Hermiticity is preserved for different k values
    let config = ExcitonConfig {
        n_kappa: 7,
        a0: 9.0,
        v0: 0.05,
        k: vec![0.0, 0.0],
        m_e: 0.3,
        m_h: 0.3,
        verbose: false,
        grid_size: 64,
        box_size: 3.0,
        save_dir: PathBuf::from("saved_data_lm"),
        recalculate: false,
        n_jobs: 32,
        use_gpu: false,
        batch_size: 32,
    };
    
    let exciton_grid = Exciton2DGridSystem::from_config(config);
    let exciton_ham: ExcitonicHamiltonian<Array2<f64>> = 
        ExcitonicHamiltonian::new(exciton_grid.full_ham, 3, 3);
    
    let q = Vector3D::new(0.01, 0.0, 0.0);
    let mode = BosonMode::new(q, Polarization::TE, 2);
    let photon_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
        BosonicHamiltonian::new(vec![mode]);
    
    let mut lm_ham = LMHamiltonian::new(exciton_ham, photon_ham, 0.01);
    
    // Test at k = (0.01, 0.01)
    lm_ham.initialize(Some(vec![0.01, 0.01]));
    
    let matrix_elements = &lm_ham.matrix_elements;
    
    println!("\nTesting Hermiticity at k = (0.01, 0.01):");
    
    // Test cos_mat
    for (key, matrix) in matrix_elements.cos_mat.iter() {
        assert!(
            is_hermitian(matrix, EPSILON),
            "cos_mat for key {:?} is not Hermitian at k = (0.01, 0.01)",
            key
        );
    }
    
    // Test sin_mat
    for (key, matrix) in matrix_elements.sin_mat.iter() {
        assert!(
            is_hermitian(matrix, EPSILON),
            "sin_mat for key {:?} is not Hermitian at k = (0.01, 0.01)",
            key
        );
    }
    
    // Test p_cos_mat
    for (key, matrix) in matrix_elements.p_cos_mat.iter() {
        assert!(
            is_anti_hermitian(matrix, EPSILON),
            "p_cos_mat for key {:?} is not anti-Hermitian at k = (0.01, 0.01)",
            key
        );
    }
}

#[test]
fn test_matrix_elements_diagonal_real() {
    // Test that diagonal elements of Hermitian matrices are real
    let config = ExcitonConfig {
        n_kappa: 7,
        a0: 9.0,
        v0: 0.05,
        k: vec![0.0, 0.0],
        m_e: 0.3,
        m_h: 0.3,
        verbose: false,
        grid_size: 64,
        box_size: 3.0,
        save_dir: PathBuf::from("saved_data_lm"),
        recalculate: false,
        n_jobs: 32,
        use_gpu: false,
        batch_size: 32,
    };
    
    let exciton_grid = Exciton2DGridSystem::from_config(config);
    let exciton_ham: ExcitonicHamiltonian<Array2<f64>> = 
        ExcitonicHamiltonian::new(exciton_grid.full_ham, 3, 3);
    
    let q = Vector3D::new(0.01, 0.0, 0.0);
    let mode = BosonMode::new(q, Polarization::TE, 2);
    let photon_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
        BosonicHamiltonian::new(vec![mode]);
    
    let mut lm_ham = LMHamiltonian::new(exciton_ham, photon_ham, 0.01);
    lm_ham.initialize(None);
    
    let matrix_elements = &lm_ham.matrix_elements;
    
    println!("\nTesting that diagonal elements are real:");
    
    // For Hermitian matrices (cos_mat, sin_mat), diagonal should be real
    for (key, matrix) in matrix_elements.cos_mat.iter() {
        for i in 0..matrix.nrows() {
            let diag_elem = matrix[[i, i]];
            assert!(
                diag_elem.im.abs() < EPSILON,
                "cos_mat[{:?}] diagonal element ({}, {}) has imaginary part: {}",
                key, i, i, diag_elem.im
            );
        }
    }
    
    for (key, matrix) in matrix_elements.sin_mat.iter() {
        for i in 0..matrix.nrows() {
            let diag_elem = matrix[[i, i]];
            assert!(
                diag_elem.im.abs() < EPSILON,
                "sin_mat[{:?}] diagonal element ({}, {}) has imaginary part: {}",
                key, i, i, diag_elem.im
            );
        }
    }
    
    // For anti-Hermitian matrices (p_cos_mat), diagonal should be purely imaginary
    for (key, matrix) in matrix_elements.p_cos_mat.iter() {
        for i in 0..matrix.nrows() {
            let diag_elem = matrix[[i, i]];
            assert!(
                diag_elem.re.abs() < EPSILON,
                "p_cos_mat[{:?}] diagonal element ({}, {}) has real part: {}",
                key, i, i, diag_elem.re
            );
        }
    }
}

#[test]
fn test_matrix_elements_size_consistency() {
    // Test that all matrix elements have consistent dimensions
    let config = ExcitonConfig {
        n_kappa: 7,
        a0: 9.0,
        v0: 0.05,
        k: vec![0.0, 0.0],
        m_e: 0.3,
        m_h: 0.3,
        verbose: false,
        grid_size: 64,
        box_size: 3.0,
        save_dir: PathBuf::from("saved_data_lm"),
        recalculate: false,
        n_jobs: 32,
        use_gpu: false,
        batch_size: 32,
    };
    
    let exciton_grid = Exciton2DGridSystem::from_config(config);
    let n_ex_eig = 3;
    let exciton_ham: ExcitonicHamiltonian<Array2<f64>> = 
        ExcitonicHamiltonian::new(exciton_grid.full_ham, n_ex_eig, 3);
    
    let q = Vector3D::new(0.01, 0.0, 0.0);
    let mode = BosonMode::new(q, Polarization::TE, 2);
    let photon_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
        BosonicHamiltonian::new(vec![mode]);
    
    let mut lm_ham = LMHamiltonian::new(exciton_ham, photon_ham, 0.01);
    lm_ham.initialize(None);
    
    let matrix_elements = &lm_ham.matrix_elements;
    
    // All matrices should be n_ex_eig x n_ex_eig
    for (key, matrix) in matrix_elements.cos_mat.iter() {
        assert_eq!(
            matrix.dim(),
            (n_ex_eig, n_ex_eig),
            "cos_mat[{:?}] has wrong dimensions",
            key
        );
    }
    
    for (key, matrix) in matrix_elements.sin_mat.iter() {
        assert_eq!(
            matrix.dim(),
            (n_ex_eig, n_ex_eig),
            "sin_mat[{:?}] has wrong dimensions",
            key
        );
    }
    
    for (key, matrix) in matrix_elements.p_cos_mat.iter() {
        assert_eq!(
            matrix.dim(),
            (n_ex_eig, n_ex_eig),
            "p_cos_mat[{:?}] has wrong dimensions",
            key
        );
    }
}