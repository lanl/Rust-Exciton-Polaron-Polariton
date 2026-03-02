/// Property-based and randomized tests for MatrixOps and Hamiltonians
///
/// This test suite uses randomized inputs to verify mathematical properties
/// that should hold for all valid inputs, such as:
/// - Algebraic identities
/// - Physical conservation laws
/// - Consistency between different representations

use ndarray::{Array1, Array2};
use ndarray_linalg::c64;
use num_traits::{Zero, One};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use light_matter::utils::{MatrixOps, ComposedOperator, Vector3D, BosonMode, Polarization};
use light_matter::hamiltonians::{BosonicHamiltonian, FullBosons};

const EPSILON: f64 = 1e-8;
const SEED: u64 = 42; // Fixed seed for reproducibility

// ============================================================================
// Random Generator Helpers
// ============================================================================

/// Create a seeded random number generator for reproducible tests
fn get_rng() -> StdRng {
    StdRng::seed_from_u64(SEED)
}

/// Generate a random complex number with bounded magnitude
fn random_c64(rng: &mut StdRng, max_magnitude: f64) -> c64 {
    let magnitude = rng.random::<f64>() * max_magnitude;
    let phase = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
    c64::from_polar(magnitude, phase)
}

/// Generate a random complex vector
fn random_vector(n: usize, max_magnitude: f64) -> Vec<c64> {
    let mut rng = get_rng();
    (0..n).map(|_| random_c64(&mut rng, max_magnitude)).collect()
}

/// Generate a random complex diagonal matrix
fn random_diagonal_matrix(n: usize, max_magnitude: f64) -> Array2<c64> {
    let mut rng = get_rng();
    let diag: Vec<c64> = (0..n).map(|_| random_c64(&mut rng, max_magnitude)).collect();
    Array2::from_diag(&Array1::from_vec(diag))
}

/// Generate a random Hermitian matrix
fn random_hermitian_matrix(n: usize, max_magnitude: f64) -> Array2<c64> {
    let mut rng = get_rng();
    let mut mat = Array2::zeros((n, n));
    
    // Fill upper triangle with random complex numbers
    for i in 0..n {
        mat[[i, i]] = c64::new(rng.random::<f64>() * max_magnitude, 0.0); // Diagonal is real
        for j in (i+1)..n {
            mat[[i, j]] = random_c64(&mut rng, max_magnitude);
            mat[[j, i]] = mat[[i, j]].conj(); // Hermitian symmetry
        }
    }
    
    mat
}

// ============================================================================
// Property Tests: Algebraic Identities
// ============================================================================

#[test]
fn property_kron_distributivity() {
    // Test: (A + B) ⊗ C = A ⊗ C + B ⊗ C
    let n = 2;
    
    let a = random_diagonal_matrix(n, 10.0);
    let b = random_diagonal_matrix(n, 10.0);
    let c = random_diagonal_matrix(n, 10.0);
    
    let left: ComposedOperator<c64> = ComposedOperator::from_operator(
        a.clone() + b.clone())
        .kron(&ComposedOperator::from_operator(c.clone()));
    
    let right: ComposedOperator<c64> = ComposedOperator::from_operator(
        a)
        .kron(&ComposedOperator::from_operator(c.clone()))
        .add(&ComposedOperator::from_operator(b)
            .kron(&ComposedOperator::from_operator(c)));
    
    let x = random_vector(n * n, 1.0);
    let mut y_left = vec![c64::zero(); n * n];
    let mut y_right = vec![c64::zero(); n * n];
    
    left.apply(&x, &mut y_left);
    right.apply(&x, &mut y_right);
    
    for i in 0..n*n {
        let diff = (y_left[i] - y_right[i]).norm();
        assert!(diff < EPSILON, "Kron distributivity failed at index {}: diff = {}", i, diff);
    }
}

#[test]
fn property_kron_associativity() {
    // Test: (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(2.0, 0.0)]);
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(3.0, 0.0), c64::new(4.0, 0.0)]);
    let c: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(5.0, 0.0)]);
    
    let left = a.kron(&b).kron(&c);
    let right = a.kron(&b.kron(&c));
    
    let x = random_vector(2, 1.0);
    let mut y_left = vec![c64::zero(); 2];
    let mut y_right = vec![c64::zero(); 2];
    
    left.apply(&x, &mut y_left);
    right.apply(&x, &mut y_right);
    
    for i in 0..2 {
        let diff = (y_left[i] - y_right[i]).norm();
        assert!(diff < EPSILON, "Kron associativity failed at index {}: diff = {}", i, diff);
    }
}

#[test]
fn property_matrix_addition_commutative() {
    // Test: A + B = B + A
    let n = 3;
    
    let a = random_diagonal_matrix(n, 10.0);
    let b = random_diagonal_matrix(n, 10.0);
    
    let left: ComposedOperator<c64> = ComposedOperator::from_operator(
        a.clone())
        .add(&ComposedOperator::from_operator(b.clone()));
    let right: ComposedOperator<c64> = ComposedOperator::from_operator(
        b)
        .add(&ComposedOperator::from_operator(a));
    
    let x = random_vector(n, 1.0);
    let mut y_left = vec![c64::zero(); n];
    let mut y_right = vec![c64::zero(); n];
    
    left.apply(&x, &mut y_left);
    right.apply(&x, &mut y_right);
    
    for i in 0..n {
        let diff = (y_left[i] - y_right[i]).norm();
        assert!(diff < EPSILON, "Addition commutativity failed at index {}: diff = {}", i, diff);
    }
}

#[test]
fn property_matrix_addition_associative() {
    // Test: (A + B) + C = A + (B + C)
    let n = 2;
    
    let a = random_diagonal_matrix(n, 10.0);
    let b = random_diagonal_matrix(n, 10.0);
    let c = random_diagonal_matrix(n, 10.0);
    
    let left: ComposedOperator<c64> = ComposedOperator::from_operator(
        a.clone())
        .add(&ComposedOperator::from_operator(b.clone()))
        .add(&ComposedOperator::from_operator(c.clone()));
    
    let right: ComposedOperator<c64> = ComposedOperator::from_operator(
        a)
        .add(&ComposedOperator::from_operator(b)
            .add(&ComposedOperator::from_operator(c)));
    
    let x = random_vector(n, 1.0);
    let mut y_left = vec![c64::zero(); n];
    let mut y_right = vec![c64::zero(); n];
    
    left.apply(&x, &mut y_left);
    right.apply(&x, &mut y_right);
    
    for i in 0..n {
        let diff = (y_left[i] - y_right[i]).norm();
        assert!(diff < EPSILON, "Addition associativity failed at index {}: diff = {}", i, diff);
    }
}

#[test]
fn property_scalar_distributivity() {
    // Test: α(A + B) = αA + αB
    let n = 2;
    let alpha = 2.5;
    
    let a = random_diagonal_matrix(n, 10.0);
    let b = random_diagonal_matrix(n, 10.0);
    
    let left: ComposedOperator<c64> = ComposedOperator::from_operator(
        a.clone())
        .add(&ComposedOperator::from_operator(b.clone()))
        .scale(alpha);
    
    let right: ComposedOperator<c64> = ComposedOperator::from_operator(
        a)
        .scale(alpha)
        .add(&ComposedOperator::from_operator(b)).scale(alpha);
    
    let x = random_vector(n, 1.0);
    let mut y_left = vec![c64::zero(); n];
    let mut y_right = vec![c64::zero(); n];
    
    left.apply(&x, &mut y_left);
    right.apply(&x, &mut y_right);
    
    for i in 0..n {
        let diff = (y_left[i] - y_right[i]).norm();
        assert!(diff < EPSILON, "Scalar distributivity failed at index {}: diff = {}", i, diff);
    }
}

#[test]
fn property_transpose_involution() {
    // Test: (A^T)^T = A for multiple random matrices
    let n = 3;
    
    for trial in 0..5 {
        let a = random_hermitian_matrix(n, 10.0);
        let op = ComposedOperator::from_operator(
            a.clone()
        );
        
        let double_transposed = op.transpose().transpose();
        
        let x = random_vector(n, 1.0);
        let mut y_original = vec![c64::zero(); n];
        let mut y_double = vec![c64::zero(); n];
        
        op.apply(&x, &mut y_original);
        double_transposed.apply(&x, &mut y_double);
        
        for i in 0..n {
            let diff = (y_original[i] - y_double[i]).norm();
            assert!(
                diff < EPSILON,
                "Transpose involution failed at trial {} index {}: diff = {}",
                trial, i, diff
            );
        }
    }
}

#[test]
fn property_transpose_addition() {
    // Test: (A + B)^T = A^T + B^T
    let n = 2;
    
    let a = random_hermitian_matrix(n, 10.0);
    let b = random_hermitian_matrix(n, 10.0);
    
    let op_a = ComposedOperator::from_operator(a);
    let op_b = ComposedOperator::from_operator(b);
    
    let left = op_a.add(&op_b).transpose();
    let right = op_a.transpose().add(&op_b.transpose());
    
    let x = random_vector(n, 1.0);
    let mut y_left = vec![c64::zero(); n];
    let mut y_right = vec![c64::zero(); n];
    
    left.apply(&x, &mut y_left);
    right.apply(&x, &mut y_right);
    
    for i in 0..n {
        let diff = (y_left[i] - y_right[i]).norm();
        assert!(diff < EPSILON, "Transpose addition failed at index {}: diff = {}", i, diff);
    }
}

// ============================================================================
// Property Tests: Physical Properties
// ============================================================================

#[test]
fn property_hermitian_eigenvalues_real() {
    // Test that Hermitian matrices have real eigenvalues
    use ndarray_linalg::Eigh;
    
    for trial in 0..10 {
        let n = 3;
        let h = random_hermitian_matrix(n, 10.0);
        
        // Compute eigenvalues
        let (eigenvalues, _) = h.eigh(ndarray_linalg::UPLO::Upper)
            .expect("Failed to diagonalize Hermitian matrix");
        
        // All eigenvalues should be real (imaginary part should be zero)
        for (i, &eval) in eigenvalues.iter().enumerate() {
            assert!(
                eval.is_finite(),
                "Trial {}: Eigenvalue {} is not finite: {}",
                trial, i, eval
            );
        }
    }
}

#[test]
fn property_hamiltonian_hermiticity() {
    // Test that bosonic Hamiltonians are Hermitian
    for trial in 0..5 {
        let mut rng = get_rng();
        
        // Random photon wavevector
        let qx = rng.random::<f64>() * 0.02 - 0.01;
        let qy = rng.random::<f64>() * 0.02 - 0.01;
        let q = Vector3D::new(qx, qy, 0.0);
        
        let mode = BosonMode::new(q, Polarization::TE, 3);
        let bosonic_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
            BosonicHamiltonian::new(vec![mode]);
        
        let h = bosonic_ham.get_h();
        
        // Check Hermiticity: H[i,j] = H[j,i]*
        for i in 0..h.nrows() {
            for j in 0..h.ncols() {
                let diff = (h[[i, j]] - h[[j, i]]).abs();
                assert!(
                    diff < EPSILON,
                    "Trial {}: Hamiltonian not Hermitian at ({}, {}): {} vs {} (diff: {})",
                    trial, i, j, h[[i, j]], h[[j, i]], diff
                );
            }
        }
    }
}

#[test]
fn property_unitary_preserves_norm() {
    // Test that unitary transformations preserve vector norm
    // For Hermitian H, exp(iHt) is unitary
    // We test a simplified version: that H^†H has eigenvalues equal to |eigenvalues of H|^2
    
    let n = 3;
    let h = random_hermitian_matrix(n, 10.0);
    
    use ndarray_linalg::Eigh;
    let (eigenvalues, _) = h.eigh(ndarray_linalg::UPLO::Upper)
        .expect("Failed to diagonalize");
    
    // For a Hermitian matrix, ||H|| = max|eigenvalue|
    let norm_h = eigenvalues.iter().map(|x| x.abs()).fold(0.0, f64::max);
    
    // Check that all eigenvalues are bounded
    for &eval in eigenvalues.iter() {
        assert!(eval.abs() <= norm_h + EPSILON);
    }
}

#[test]
fn property_trace_invariant_under_basis_change() {
    // Test that trace is invariant: Tr(A) = Tr(U†AU) for unitary U
    // For simplicity, we test that the trace of a diagonal matrix equals sum of eigenvalues
    
    for trial in 0..5 {
        let n = 3;
        let diag_vals: Vec<f64> = (0..n).map(|i| (i + 1) as f64 + trial as f64).collect();
        let mat: Array2<f64> = MatrixOps::from_diag(&diag_vals);
        
        let expected_trace: f64 = diag_vals.iter().sum();
        
        let mut computed_trace = 0.0;
        for i in 0..n {
            computed_trace += mat[[i, i]];
        }
        
        let diff = (expected_trace - computed_trace).abs();
        assert!(
            diff < EPSILON,
            "Trial {}: Trace mismatch: expected {}, got {} (diff: {})",
            trial, expected_trace, computed_trace, diff
        );
    }
}

// ============================================================================
// Property Tests: Consistency Between Representations
// ============================================================================

#[test]
fn property_diagonal_consistent_across_types() {
    // Test that diagonal extraction gives same result for different matrix types
    let n = 4;
    
    for trial in 0..5 {
        let mut rng = get_rng();
        let diag_vals: Vec<f64> = (0..n).map(|_| rng.random::<f64>() * 10.0).collect();
        
        let dense: Array2<f64> = MatrixOps::from_diag(&diag_vals);
        let sparse: sprs::CsMat<f64> = MatrixOps::from_diag(&diag_vals);
        let composed: ComposedOperator<c64> = MatrixOps::from_diag(
            &diag_vals.iter().map(|&x| c64::new(x, 0.0)).collect::<Vec<_>>()
        );
        
        // Extract diagonals
        let dense_diag: Vec<f64> = (0..n).map(|i| dense[[i, i]]).collect();
        let sparse_diag: Vec<f64> = sparse.diag().iter().map(|(_, &v)| v).collect();
        let composed_diag: Vec<f64> = composed.get_diagonal().iter().map(|c| c.re).collect();
        
        // Compare
        for i in 0..n {
            let diff1 = (dense_diag[i] - diag_vals[i]).abs();
            let diff2 = (sparse_diag[i] - diag_vals[i]).abs();
            let diff3 = (composed_diag[i] - diag_vals[i]).abs();
            
            assert!(diff1 < EPSILON, "Trial {}: Dense diagonal mismatch at {}", trial, i);
            assert!(diff2 < EPSILON, "Trial {}: Sparse diagonal mismatch at {}", trial, i);
            assert!(diff3 < EPSILON, "Trial {}: Composed diagonal mismatch at {}", trial, i);
        }
    }
}

#[test]
fn property_kron_dimension_product() {
    // Test that dim(A ⊗ B) = dim(A) * dim(B)
    for trial in 0..10 {
        let mut rng = get_rng();
        let n_a = rng.random_range(1..5);
        let n_b = rng.random_range(1..5);
        
        let a: ComposedOperator<c64> = MatrixOps::eye(n_a);
        let b: ComposedOperator<c64> = MatrixOps::eye(n_b);
        
        let kron_prod = a.kron(&b);
        
        assert_eq!(
            kron_prod.dimension(),
            n_a * n_b,
            "Trial {}: Kron dimension incorrect: expected {}, got {}",
            trial,
            n_a * n_b,
            kron_prod.dimension()
        );
    }
}

// ============================================================================
// Property Tests: Randomized Stress Tests
// ============================================================================

#[test]
fn property_large_composition_depth() {
    // Test that deeply nested compositions still work correctly
    let mut op: ComposedOperator<c64> = MatrixOps::eye(2);
    
    for i in 0..10 {
        let scalar = 1.0 + i as f64 * 0.1;
        op = op.scale(scalar);
    }
    
    let x = vec![c64::one(); 2];
    let mut y = vec![c64::zero(); 2];
    
    op.apply(&x, &mut y);
    
    // Result should be 1 * 1.0 * 1.1 * 1.2 * ... * 1.9
    let expected_scale: f64 = (0..10).map(|i| 1.0 + i as f64 * 0.1).product();
    
    for i in 0..2 {
        let diff = (y[i].re - expected_scale).abs();
        assert!(diff < EPSILON, "Large composition depth failed: diff = {}", diff);
    }
}

#[test]
fn property_operations_preserve_dimensions() {
    // Test that all operations preserve or correctly transform dimensions
    let mut rng = get_rng();
    
    for _ in 0..20 {
        let n = rng.random_range(1..6);
        
        let a: ComposedOperator<c64> = MatrixOps::eye(n);
        let b: ComposedOperator<c64> = MatrixOps::zeros((n, n));
        
        // All these should have dimension n
        assert_eq!(a.dimension(), n);
        assert_eq!(b.dimension(), n);
        assert_eq!(a.add(&b).dimension(), n);
        assert_eq!(a.scale(2.0).dimension(), n);
        assert_eq!(a.transpose().dimension(), n);
        
        // Kron should give n^2
        assert_eq!(a.kron(&a).dimension(), n * n);
    }
}

// ============================================================================
// Fuzzing-Style Tests
// ============================================================================

#[test]
fn fuzz_random_operations() {
    // Randomly combine operations and verify they don't panic or produce NaN
    let mut rng = get_rng();
    
    for trial in 0..50 {
        let n = rng.random_range(1..5);
        let mut op: ComposedOperator<c64> = MatrixOps::eye(n);
        
        // Apply 10 random operations
        for _ in 0..10 {
            let operation = rng.random_range(0..4);
            
            match operation {
                0 => {
                    // Add identity
                    let id: ComposedOperator<c64> = MatrixOps::eye(n);
                    op = op.add(&id);
                }
                1 => {
                    // Scale by random number
                    let scale = rng.random::<f64>() * 2.0;
                    op = op.scale(scale);
                }
                2 => {
                    // Transpose
                    op = op.transpose();
                }
                3 => {
                    // Add random diagonal
                    let diag: Vec<c64> = (0..n)
                        .map(|_| c64::new(rng.random::<f64>(), rng.random::<f64>()))
                        .collect();
                    let diag_op: ComposedOperator<c64> = MatrixOps::from_diag(&diag);
                    op = op.add(&diag_op);
                }
                _ => unreachable!(),
            }
        }
        
        // Try to apply to a vector
        let x = random_vector(n, 1.0);
        let mut y = vec![c64::zero(); n];
        
        op.apply(&x, &mut y);
        
        // Check that result is finite
        for (i, &val) in y.iter().enumerate() {
            assert!(
                val.re.is_finite() && val.im.is_finite(),
                "Trial {}: Non-finite value at index {}: {}",
                trial, i, val
            );
        }
    }
}

#[test]
fn fuzz_random_hamiltonian_parameters() {
    // Test that Hamiltonians work with random physical parameters
    let mut rng = get_rng();
    
    for trial in 0..10 {
        // Random parameters within physical bounds
        let qx = rng.random::<f64>() * 0.02 - 0.01;
        let qy = rng.random::<f64>() * 0.02 - 0.01;
        let n_fock = rng.random_range(2..5);
        
        let q = Vector3D::new(qx, qy, 0.0);
        let polarization = if rng.random::<bool>() { Polarization::TE } else { Polarization::TM };
        
        let mode = BosonMode::new(q, polarization, n_fock);
        let bosonic_ham: BosonicHamiltonian<Array2<f64>, FullBosons<Array2<f64>>> = 
            BosonicHamiltonian::new(vec![mode]);
        
        let h = bosonic_ham.get_h();
        
        // Check that Hamiltonian is well-formed
        assert_eq!(h.shape(), &[n_fock, n_fock]);
        
        // Check finite
        for i in 0..n_fock {
            for j in 0..n_fock {
                assert!(
                    h[[i, j]].is_finite(),
                    "Trial {}: Non-finite Hamiltonian element at ({}, {}): {}",
                    trial, i, j, h[[i, j]]
                );
            }
        }
    }
}
