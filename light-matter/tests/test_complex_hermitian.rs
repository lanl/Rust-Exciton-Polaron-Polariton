/// Test for complex Hermitian matrix diagonalization
/// 
/// This test verifies that the SLEPc complex implementation works correctly
/// by diagonalizing a simple known Hermitian matrix.

use ndarray::{Array1, Array2};
use num_complex::Complex64 as c64;
use num_traits::Zero;
use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize SLEPc once for all tests
fn initialize_slepc() {
    INIT.call_once(|| {
        rust_slepc_bindings::initialize().expect("Failed to initialize SLEPc");
    });
}

/// Simple test operator: diagonal matrix with known eigenvalues
struct DiagonalHermitianOperator {
    eigenvalues: Vec<f64>,
}

impl DiagonalHermitianOperator {
    fn new(eigenvalues: Vec<f64>) -> Self {
        Self { eigenvalues }
    }
}

impl light_matter::utils::SolverOperator<c64> for DiagonalHermitianOperator {
    fn apply(&self, x: &[c64], y: &mut [c64]) {
        // Diagonal matrix: y = D * x where D is diagonal
        for (i, (&xi, yi)) in x.iter().zip(y.iter_mut()).enumerate() {
            *yi = c64::new(self.eigenvalues[i], 0.0) * xi;
        }
    }
    
    fn apply_transpose(&self, x: &[c64], y: &mut [c64]) {
        // Hermitian, so transpose = conjugate transpose, but diagonal so same as apply
        self.apply(x, y)
    }
    
    fn get_diagonal(&self) -> Vec<c64> {
        self.eigenvalues.iter().map(|&e| c64::new(e, 0.0)).collect()
    }
    
    fn dimension(&self) -> usize {
        self.eigenvalues.len()
    }
    
    fn diagonalize(
        &self,
        _n_eig: Option<usize>,
        _solve_type: Option<rust_slepc_bindings::EPSType>,
        _initial_guess: Option<&Array2<c64>>,
    ) -> (Array1<f64>, Array2<c64>) {
        unimplemented!("Use diagonalize_slepc instead")
    }
}

/// More complex test: 2x2 Pauli matrix (Hermitian)
struct PauliXOperator;

impl light_matter::utils::SolverOperator<c64> for PauliXOperator {
    fn apply(&self, x: &[c64], y: &mut [c64]) {
        // Pauli X matrix: [[0, 1], [1, 0]]
        // Eigenvalues: +1, -1
        y[0] = x[1];
        y[1] = x[0];
    }
    
    fn apply_transpose(&self, x: &[c64], y: &mut [c64]) {
        self.apply(x, y) // Hermitian
    }
    
    fn get_diagonal(&self) -> Vec<c64> {
        vec![c64::zero(), c64::zero()]
    }
    
    fn dimension(&self) -> usize {
        2
    }
    
    fn diagonalize(
        &self,
        _n_eig: Option<usize>,
        _solve_type: Option<rust_slepc_bindings::EPSType>,
        _initial_guess: Option<&Array2<c64>>,
    ) -> (Array1<f64>, Array2<c64>) {
        unimplemented!("Use diagonalize_slepc instead")
    }
}

#[test]
fn test_diagonal_hermitian() {
    use light_matter::slepc::diagonalize_slepc;
    use rust_slepc_bindings::EPSType;
    
    // Initialize SLEPc (only once)
    initialize_slepc();
    
    // Create a diagonal operator with known eigenvalues
    let eigenvalues = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let op = DiagonalHermitianOperator::new(eigenvalues.clone());
    
    // Diagonalize
    let (computed_evals, _evecs) = diagonalize_slepc(
        &op,
        5,
        32,
        Some(EPSType::KrylovSchur),
        None,
    );
    
    // Check eigenvalues
    println!("Expected eigenvalues: {:?}", eigenvalues);
    println!("Computed eigenvalues: {:?}", computed_evals);
    
    for (i, (&expected, &computed)) in eigenvalues.iter().zip(computed_evals.iter()).enumerate() {
        let error = (expected - computed).abs();
        println!("Eigenvalue {}: expected={}, computed={}, error={:.2e}", 
                 i, expected, computed, error);
        assert!(error < 1e-10, "Eigenvalue {} mismatch: expected {}, got {}", 
                i, expected, computed);
    }
}

#[test]
fn test_pauli_x() {
    use light_matter::slepc::diagonalize_slepc;
    use rust_slepc_bindings::EPSType;
    
    // Initialize SLEPc (only once)
    initialize_slepc();
    
    // Create Pauli X operator
    let op = PauliXOperator;
    
    // Diagonalize
    let (computed_evals, _evecs) = diagonalize_slepc(
        &op,
        2,
        32,
        Some(EPSType::KrylovSchur),
        None,
    );
    
    // Pauli X has eigenvalues +1 and -1
    let mut sorted_evals = computed_evals.to_vec();
    sorted_evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    println!("Computed eigenvalues: {:?}", sorted_evals);
    
    assert!((sorted_evals[0] - (-1.0)).abs() < 1e-10, 
            "First eigenvalue should be -1, got {}", sorted_evals[0]);
    assert!((sorted_evals[1] - 1.0).abs() < 1e-10, 
            "Second eigenvalue should be 1, got {}", sorted_evals[1]);
}

#[test]
fn test_hermiticity_check() {
    use light_matter::slepc::verify_hermitian;
    use light_matter::utils::SolverOperator;
    
    // First, manually test the hermiticity condition for a simple case
    let eigenvalues = vec![1.0, 2.0];
    let op = DiagonalHermitianOperator::new(eigenvalues.clone());
    
    // Create simple test vectors
    let x = vec![c64::new(1.0, 0.5), c64::new(0.3, -0.2)];
    let y = vec![c64::new(0.7, 0.1), c64::new(-0.4, 0.6)];
    
    // Compute H*x and H*y
    let mut hx = vec![c64::zero(); 2];
    let mut hy = vec![c64::zero(); 2];
    op.apply(&x, &mut hx);
    op.apply(&y, &mut hy);
    
    println!("Manual hermiticity check:");
    println!("x = {:?}", x);
    println!("y = {:?}", y);
    println!("H*x = {:?}", hx);
    println!("H*y = {:?}", hy);
    
    // Compute <y, H*x>
    let y_hx: c64 = y.iter().zip(hx.iter()).map(|(yi, hxi)| yi.conj() * hxi).sum();
    println!("<y, H*x> = {}", y_hx);
    
    // Compute <H*y, x>
    let hy_x: c64 = hy.iter().zip(x.iter()).map(|(hyi, xi)| hyi.conj() * xi).sum();
    println!("<H*y, x> = {}", hy_x);
    
    // For Hermitian: <y, H*x> should equal <H*y, x>
    let diff = (y_hx - hy_x).norm();
    println!("Difference: {}", y_hx - hy_x);
    println!("Difference norm: {:.2e}", diff);
    println!();
    
    assert!(diff < 1e-10, "Manual check: Diagonal operator should be Hermitian, but diff = {:.2e}", diff);
    
    // Now test with the verify_hermitian function
    let eigenvalues = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let op = DiagonalHermitianOperator::new(eigenvalues);
    
    // Diagonal matrices with real entries are Hermitian
    // Use 1e-8 tolerance to account for numerical errors
    assert!(verify_hermitian(&op, 1e-8), "Diagonal operator should be Hermitian");
    
    // Test Pauli X (should be Hermitian)
    let pauli_x = PauliXOperator;
    assert!(verify_hermitian(&pauli_x, 1e-8), "Pauli X should be Hermitian");
}

/// Test that verifies the complex matrix-vector product works correctly
#[test]
fn test_complex_matvec() {

    use light_matter::utils::SolverOperator;
    // Create a simple 2x2 Hermitian matrix
    // H = [[1, i], [-i, 2]]
    struct SimpleHermitian;
    
    impl SolverOperator<c64> for SimpleHermitian {
        fn apply(&self, x: &[c64], y: &mut [c64]) {
            // H * x where H = [[1, i], [-i, 2]]
            y[0] = x[0] + c64::new(0.0, 1.0) * x[1];
            y[1] = c64::new(0.0, -1.0) * x[0] + c64::new(2.0, 0.0) * x[1];
        }
        
        fn apply_transpose(&self, x: &[c64], y: &mut [c64]) {
            self.apply(x, y) // Hermitian
        }
        
        fn get_diagonal(&self) -> Vec<c64> {
            vec![c64::new(1.0, 0.0), c64::new(2.0, 0.0)]
        }
        
        fn dimension(&self) -> usize {
            2
        }
        
        fn diagonalize(
            &self,
            _n_eig: Option<usize>,
            _solve_type: Option<rust_slepc_bindings::EPSType>,
            _initial_guess: Option<&Array2<c64>>,
        ) -> (Array1<f64>, Array2<c64>) {
            unimplemented!()
        }
    }
    
    let op = SimpleHermitian;
    
    // Test with a simple vector [1, 0]
    let x = vec![c64::new(1.0, 0.0), c64::new(0.0, 0.0)];
    let mut y = vec![c64::zero(); 2];
    op.apply(&x, &mut y);
    
    // Expected: H * [1, 0] = [1, -i]
    assert!((y[0] - c64::new(1.0, 0.0)).norm() < 1e-10);
    assert!((y[1] - c64::new(0.0, -1.0)).norm() < 1e-10);
    
    // Test with [0, 1]
    let x = vec![c64::new(0.0, 0.0), c64::new(1.0, 0.0)];
    let mut y = vec![c64::zero(); 2];
    op.apply(&x, &mut y);
    
    // Expected: H * [0, 1] = [i, 2]
    assert!((y[0] - c64::new(0.0, 1.0)).norm() < 1e-10);
    assert!((y[1] - c64::new(2.0, 0.0)).norm() < 1e-10);
    
    // Verify hermiticity manually
    let x = vec![c64::new(1.0, 0.5), c64::new(0.3, -0.2)];
    let y = vec![c64::new(0.7, 0.1), c64::new(-0.4, 0.6)];
    
    let mut hx = vec![c64::zero(); 2];
    let mut hy = vec![c64::zero(); 2];
    op.apply(&x, &mut hx);
    op.apply(&y, &mut hy);
    
    let y_hx: c64 = y.iter().zip(hx.iter()).map(|(yi, hxi)| yi.conj() * hxi).sum();
    let hy_x: c64 = hy.iter().zip(x.iter()).map(|(hyi, xi)| hyi.conj() * xi).sum();
    
    println!("SimpleHermitian hermiticity check:");
    println!("<y, H*x> = {}", y_hx);
    println!("<H*y, x> = {}", hy_x);
    println!("Difference: {}", y_hx - hy_x);
    
    let diff = (y_hx - hy_x).norm();
    assert!(diff < 1e-10, "SimpleHermitian should be Hermitian, but diff = {:.2e}", diff);
}
