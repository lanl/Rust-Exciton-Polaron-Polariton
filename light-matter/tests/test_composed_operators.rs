/// Tests for ComposedOperator advanced features
///
/// This test suite focuses on:
/// - Operator composition and optimization
/// - Matrix-free operations
/// - SolverOperator trait implementation
/// - Diagonal extraction
/// - Transpose operations

use ndarray::{arr2};
use ndarray_linalg::c64;
use num_traits::{Zero, One};
use light_matter::utils::{
    ComposedOperator, 
    MatrixOps, SolverOperator,
};
use sprs::CsMat;

const EPSILON: f64 = 1e-10;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a simple test vector
fn test_vector(n: usize) -> Vec<c64> {
    (0..n).map(|i| c64::new((i + 1) as f64, 0.0)).collect()
}

/// Check if two vectors are approximately equal
fn assert_vec_close(a: &[c64], b: &[c64], tol: f64, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: Length mismatch", msg);
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).norm();
        assert!(
            diff < tol,
            "{}: Vectors differ at index {}: {} vs {} (diff: {})",
            msg, i, a[i], b[i], diff
        );
    }
}

// ============================================================================
// Basic ComposedOperator Tests
// ============================================================================

#[test]
fn test_composed_zero_operator() {
    let zero: ComposedOperator<c64> = MatrixOps::zeros((3, 3));
    
    assert_eq!(zero.dimension(), 3);
    
    let x = test_vector(3);
    let mut y = vec![c64::zero(); 3];
    
    zero.apply(&x, &mut y);
    
    for i in 0..3 {
        assert!(y[i].norm() < EPSILON);
    }
}

#[test]
fn test_composed_identity_operator() {
    let id: ComposedOperator<c64> = MatrixOps::eye(4);
    
    assert_eq!(id.dimension(), 4);
    
    let x = test_vector(4);
    let mut y = vec![c64::zero(); 4];
    
    id.apply(&x, &mut y);
    
    assert_vec_close(&x, &y, EPSILON, "Identity should preserve vector");
}

#[test]
fn test_composed_diagonal_extraction() {
    let diag_vals = vec![c64::new(1.0, 0.0), c64::new(2.0, 0.0), c64::new(3.0, 0.0)];
    let op: ComposedOperator<c64> = MatrixOps::from_diag(&diag_vals);
    
    let extracted_diag = op.get_diagonal();
    
    assert_vec_close(&diag_vals, &extracted_diag, EPSILON, "Diagonal extraction");
}

// ============================================================================
// Operator Composition Tests
// ============================================================================

#[test]
fn test_add_composition() {
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[
        c64::new(1.0, 0.0),
        c64::new(2.0, 0.0),
        c64::new(3.0, 0.0),
    ]);
    
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[
        c64::new(4.0, 0.0),
        c64::new(5.0, 0.0),
        c64::new(6.0, 0.0),
    ]);
    
    let sum = a.add(&b);
    
    let x = vec![c64::one(); 3];
    let mut y = vec![c64::zero(); 3];
    
    sum.apply(&x, &mut y);
    
    // (1+4)*1, (2+5)*1, (3+6)*1
    let expected = vec![c64::new(5.0, 0.0), c64::new(7.0, 0.0), c64::new(9.0, 0.0)];
    assert_vec_close(&y, &expected, EPSILON, "Addition composition");
}

#[test]
fn test_scalar_mul_composition() {
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[
        c64::new(1.0, 0.0),
        c64::new(2.0, 0.0),
    ]);
    
    let scaled = a.scale(3.0);
    
    let x = vec![c64::one(); 2];
    let mut y = vec![c64::zero(); 2];
    
    scaled.apply(&x, &mut y);
    
    let expected = vec![c64::new(3.0, 0.0), c64::new(6.0, 0.0)];
    assert_vec_close(&y, &expected, EPSILON, "Scalar multiplication");
}

#[test]
fn test_dot_composition() {
    // Two diagonal matrices
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[
        c64::new(2.0, 0.0),
        c64::new(3.0, 0.0),
    ]);
    
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[
        c64::new(4.0, 0.0),
        c64::new(5.0, 0.0),
    ]);
    
    let product = a.dot(&b);
    
    let x = vec![c64::one(); 2];
    let mut y = vec![c64::zero(); 2];
    
    product.apply(&x, &mut y);
    
    // AB = diag(2*4, 3*5) = diag(8, 15)
    let expected = vec![c64::new(8.0, 0.0), c64::new(15.0, 0.0)];
    assert_vec_close(&y, &expected, EPSILON, "Dot product composition");
}

#[test]
fn test_kron_composition_diagonal() {
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[
        c64::new(2.0, 0.0),
        c64::new(3.0, 0.0),
    ]);
    
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[
        c64::new(5.0, 0.0),
        c64::new(7.0, 0.0),
    ]);
    
    let kron_prod = a.kron(&b);
    
    assert_eq!(kron_prod.dimension(), 4);
    
    // For diagonal matrices, kron product is also diagonal
    let diag = kron_prod.get_diagonal();
    
    // [2, 3] ⊗ [5, 7] = [2*5, 2*7, 3*5, 3*7] = [10, 14, 15, 21]
    let expected = vec![
        c64::new(10.0, 0.0),
        c64::new(14.0, 0.0),
        c64::new(15.0, 0.0),
        c64::new(21.0, 0.0),
    ];
    assert_vec_close(&diag, &expected, EPSILON, "Kronecker product diagonal");
}

#[test]
fn test_kron_with_identity() {
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(2.0, 0.0), c64::new(3.0, 0.0)]);
    let id: ComposedOperator<c64> = MatrixOps::eye(2);
    
    // A ⊗ I
    let result = a.kron(&id);
    
    let x = vec![c64::one(); 4];
    let mut y = vec![c64::zero(); 4];
    
    result.apply(&x, &mut y);
    
    // A ⊗ I acting on [1,1,1,1] should give [2,2,3,3]
    let expected = vec![
        c64::new(2.0, 0.0),
        c64::new(2.0, 0.0),
        c64::new(3.0, 0.0),
        c64::new(3.0, 0.0),
    ];
    assert_vec_close(&y, &expected, EPSILON, "Kronecker with identity");
}

#[test]
fn test_identity_kron_a() {
    let id: ComposedOperator<c64> = MatrixOps::eye(2);
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(5.0, 0.0), c64::new(7.0, 0.0)]);
    
    // I ⊗ A
    let result = id.kron(&a);
    
    let x = vec![c64::one(); 4];
    let mut y = vec![c64::zero(); 4];
    
    result.apply(&x, &mut y);
    
    // I ⊗ A acting on [1,1,1,1] should give [5,7,5,7]
    let expected = vec![
        c64::new(5.0, 0.0),
        c64::new(7.0, 0.0),
        c64::new(5.0, 0.0),
        c64::new(7.0, 0.0),
    ];
    assert_vec_close(&y, &expected, EPSILON, "Identity kronecker A");
}

// ============================================================================
// Transpose Tests
// ============================================================================

#[test]
fn test_transpose_diagonal() {
    let diag_vals = vec![c64::new(1.0, 1.0), c64::new(2.0, -1.0)];
    let op: ComposedOperator<c64> = MatrixOps::from_diag::<c64>(&diag_vals);
    
    let transposed = op.transpose();
    
    let x = test_vector(2);
    let mut y1 = vec![c64::zero(); 2];
    let mut y2 = vec![c64::zero(); 2];
    
    op.apply(&x, &mut y1);
    transposed.apply(&x, &mut y2);
    
    // For diagonal matrices, transpose has no effect
    assert_vec_close(&y1, &y2, EPSILON, "Diagonal transpose");
}

#[test]
fn test_transpose_general() {
    // Create a non-symmetric matrix
    let dense = arr2(&[
        [c64::new(1.0, 0.0), c64::new(2.0, 0.0)],
        [c64::new(3.0, 0.0), c64::new(4.0, 0.0)],
    ]);
    let op = ComposedOperator::from_operator(dense);
    
    let transposed = op.transpose();
    
    let x = vec![c64::new(1.0, 0.0), c64::new(0.0, 0.0)];
    let mut y = vec![c64::zero(); 2];
    
    transposed.apply(&x, &mut y);
    
    // Transpose([[1,2],[3,4]]) * [1,0] = [[1,3],[2,4]] * [1,0] = [1, 2]
    let expected = vec![c64::new(1.0, 0.0), c64::new(2.0, 0.0)];
    assert_vec_close(&y, &expected, EPSILON, "General transpose");
}

#[test]
fn test_transpose_involution() {
    // (A^T)^T = A
    let dense = arr2(&[
        [c64::new(1.0, 0.0), c64::new(2.0, 1.0)],
        [c64::new(3.0, -1.0), c64::new(4.0, 0.0)],
    ]);
    let op = ComposedOperator::from_operator(dense);
    
    let double_transposed = op.transpose().transpose();
    
    let x = test_vector(2);
    let mut y1 = vec![c64::zero(); 2];
    let mut y2 = vec![c64::zero(); 2];
    
    op.apply(&x, &mut y1);
    double_transposed.apply(&x, &mut y2);
    
    assert_vec_close(&y1, &y2, EPSILON, "Double transpose should equal original");
}

#[test]
fn test_transpose_add() {
    // (A + B)^T = A^T + B^T
    let a = ComposedOperator::from_operator(arr2(&[
        [c64::new(1.0, 0.0), c64::new(2.0, 0.0)],
        [c64::new(3.0, 0.0), c64::new(4.0, 0.0)],
    ]));
    
    let b = ComposedOperator::from_operator(arr2(&[
        [c64::new(5.0, 0.0), c64::new(6.0, 0.0)],
        [c64::new(7.0, 0.0), c64::new(8.0, 0.0)],
    ]));
    
    let sum_transposed = a.add(&b).transpose();
    let transposed_sum = a.transpose().add(&b.transpose());
    
    let x = test_vector(2);
    let mut y1 = vec![c64::zero(); 2];
    let mut y2 = vec![c64::zero(); 2];
    
    sum_transposed.apply(&x, &mut y1);
    transposed_sum.apply(&x, &mut y2);
    
    assert_vec_close(&y1, &y2, EPSILON, "(A+B)^T = A^T + B^T");
}

#[test]
fn test_apply_transpose_method() {
    let dense = arr2(&[
        [c64::new(1.0, 0.0), c64::new(2.0, 0.0)],
        [c64::new(3.0, 0.0), c64::new(4.0, 0.0)],
    ]);
    let op = ComposedOperator::from_operator(dense);
    
    let x = vec![c64::new(1.0, 0.0), c64::new(0.0, 0.0)];
    let mut y1 = vec![c64::zero(); 2];
    let mut y2 = vec![c64::zero(); 2];
    
    // Use transpose() wrapper
    let transposed = op.transpose();
    transposed.apply(&x, &mut y1);
    
    // Use apply_transpose directly
    op.apply_transpose(&x, &mut y2);
    
    assert_vec_close(&y1, &y2, EPSILON, "apply_transpose should match transpose().apply()");
}

// ============================================================================
// Diagonal Extraction Tests
// ============================================================================

#[test]
fn test_diagonal_add() {
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(1.0, 0.0), c64::new(2.0, 0.0)]);
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(3.0, 0.0), c64::new(4.0, 0.0)]);
    
    let sum = a.add(&b);
    let diag = sum.get_diagonal();
    
    let expected = vec![c64::new(4.0, 0.0), c64::new(6.0, 0.0)];
    assert_vec_close(&diag, &expected, EPSILON, "Diagonal of sum");
}

#[test]
fn test_diagonal_dot_diagonal() {
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(2.0, 0.0), c64::new(3.0, 0.0)]);
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(4.0, 0.0), c64::new(5.0, 0.0)]);
    
    let product = a.dot(&b);
    let diag = product.get_diagonal();
    
    // Product of diagonal matrices is diagonal
    let expected = vec![c64::new(8.0, 0.0), c64::new(15.0, 0.0)];
    assert_vec_close(&diag, &expected, EPSILON, "Diagonal of product");
}

#[test]
fn test_diagonal_kron() {
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(2.0, 0.0), c64::new(3.0, 0.0)]);
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(5.0, 0.0), c64::new(7.0, 0.0)]);
    
    let kron_prod = a.kron(&b);
    let diag = kron_prod.get_diagonal();
    
    let expected = vec![
        c64::new(10.0, 0.0),
        c64::new(14.0, 0.0),
        c64::new(15.0, 0.0),
        c64::new(21.0, 0.0),
    ];
    assert_vec_close(&diag, &expected, EPSILON, "Diagonal of Kronecker product");
}

#[test]
fn test_diagonal_identity() {
    let id: ComposedOperator<c64> = MatrixOps::eye(3);
    let diag = id.get_diagonal();
    
    let expected = vec![c64::one(); 3];
    assert_vec_close(&diag, &expected, EPSILON, "Diagonal of identity");
}

#[test]
fn test_diagonal_zero() {
    let zero: ComposedOperator<c64> = MatrixOps::zeros((3, 3));
    let diag = zero.get_diagonal();
    
    let expected = vec![c64::zero(); 3];
    assert_vec_close(&diag, &expected, EPSILON, "Diagonal of zero");
}

// ============================================================================
// Complex Composition Tests
// ============================================================================

#[test]
fn test_nested_composition() {
    // ((A + B) ⊗ C) * scalar
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(1.0, 0.0)]);
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(2.0, 0.0)]);
    let c: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(3.0, 0.0), c64::new(4.0, 0.0)]);
    
    let result = a.add(&b).kron(&c).scale(2.0);
    
    let x = vec![c64::one(); 2];
    let mut y = vec![c64::zero(); 2];
    
    result.apply(&x, &mut y);
    
    // (1+2) ⊗ [3,4] * 2 = 3 ⊗ [3,4] * 2 = [9, 12] * 2 = [18, 24]
    let expected = vec![c64::new(18.0, 0.0), c64::new(24.0, 0.0)];
    assert_vec_close(&y, &expected, EPSILON, "Nested composition");
}

#[test]
fn test_associativity_add() {
    // (A + B) + C = A + (B + C)
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(1.0, 0.0)]);
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(2.0, 0.0)]);
    let c: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(3.0, 0.0)]);
    
    let left = a.add(&b).add(&c);
    let right = a.add(&b.add(&c));
    
    let x = vec![c64::one(); 1];
    let mut y1 = vec![c64::zero(); 1];
    let mut y2 = vec![c64::zero(); 1];
    
    left.apply(&x, &mut y1);
    right.apply(&x, &mut y2);
    
    assert_vec_close(&y1, &y2, EPSILON, "Addition associativity");
}

#[test]
fn test_distributivity() {
    // A * (B + C) = A*B + A*C for diagonal matrices
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(2.0, 0.0), c64::new(3.0, 0.0)]);
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(4.0, 0.0), c64::new(5.0, 0.0)]);
    let c: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(6.0, 0.0), c64::new(7.0, 0.0)]);
    
    let left = a.dot(&b.add(&c));
    let right = a.dot(&b).add(&a.dot(&c));
    
    let x = vec![c64::one(); 2];
    let mut y1 = vec![c64::zero(); 2];
    let mut y2 = vec![c64::zero(); 2];
    
    left.apply(&x, &mut y1);
    right.apply(&x, &mut y2);
    
    assert_vec_close(&y1, &y2, EPSILON, "Distributivity");
}

// ============================================================================
// SolverOperator Integration Tests
// ============================================================================

#[test]
fn test_solver_operator_dimension() {
    let op: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(1.0, 0.0), c64::new(2.0, 0.0)]);
    
    // Should implement SolverOperator
    assert_eq!(op.dimension(), 2);
}

#[test]
fn test_solver_operator_apply() {
    let op: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(3.0, 0.0), c64::new(4.0, 0.0)]);
    
    let x = vec![c64::one(); 2];
    let mut y = vec![c64::zero(); 2];
    
    SolverOperator::apply(&op, &x, &mut y);
    
    let expected = vec![c64::new(3.0, 0.0), c64::new(4.0, 0.0)];
    assert_vec_close(&y, &expected, EPSILON, "SolverOperator::apply");
}

#[test]
fn test_solver_operator_get_diagonal() {
    let diag_vals = vec![c64::new(1.0, 0.0), c64::new(2.0, 0.0), c64::new(3.0, 0.0)];
    let op: ComposedOperator<c64> = MatrixOps::from_diag(&diag_vals);
    
    let extracted = SolverOperator::get_diagonal(&op);
    
    assert_vec_close(&diag_vals, &extracted, EPSILON, "SolverOperator::get_diagonal");
}

// ============================================================================
// Performance/Stress Tests
// ============================================================================

#[test]
fn test_large_kron_structure() {
    // Test that large Kronecker products maintain correct structure
    let a: ComposedOperator<c64> = MatrixOps::eye(10);
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&vec![c64::new(2.0, 0.0); 10]);
    
    let kron_prod = a.kron(&b);
    
    assert_eq!(kron_prod.dimension(), 100);
    
    // Test on a simple vector
    let x = vec![c64::one(); 100];
    let mut y = vec![c64::zero(); 100];
    
    kron_prod.apply(&x, &mut y);
    
    // I ⊗ (2*I) should give 2*I
    for i in 0..100 {
        assert!((y[i] - c64::new(2.0, 0.0)).norm() < EPSILON);
    }
}

#[test]
fn test_deeply_nested_operations() {
    // Test deeply nested structure: (((A + B) * 2) ⊗ C) + D
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(1.0, 0.0)]);
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(1.0, 0.0)]);
    let c: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(1.0, 0.0), c64::new(1.0, 0.0)]);
    let d: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(1.0, 0.0), c64::new(1.0, 0.0)]);
    
    let result = a.add(&b).scale(2.0).kron(&c).add(&d);
    
    assert_eq!(result.dimension(), 2);
    
    let x = vec![c64::one(); 2];
    let mut y = vec![c64::zero(); 2];
    
    result.apply(&x, &mut y);
    
    // (1+1)*2 ⊗ [1,1] + [1,1] = 4*[1,1] + [1,1] = [5, 5]
    let expected = vec![c64::new(5.0, 0.0), c64::new(5.0, 0.0)];
    assert_vec_close(&y, &expected, EPSILON, "Deeply nested operations");
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_zero_times_anything() {
    let zero: ComposedOperator<c64> = MatrixOps::zeros((2, 2));
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(5.0, 0.0), c64::new(7.0, 0.0)]);
    
    let product = zero.dot(&a);
    
    let x = test_vector(2);
    let mut y = vec![c64::zero(); 2];
    
    product.apply(&x, &mut y);
    
    for i in 0..2 {
        assert!(y[i].norm() < EPSILON, "Zero times anything should be zero");
    }
}

#[test]
fn test_identity_times_anything() {
    let id: ComposedOperator<c64> = MatrixOps::eye(3);
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[
        c64::new(2.0, 0.0),
        c64::new(3.0, 0.0),
        c64::new(4.0, 0.0),
    ]);
    
    let product = id.dot(&a);
    
    let x = vec![c64::one(); 3];
    let mut y1 = vec![c64::zero(); 3];
    let mut y2 = vec![c64::zero(); 3];
    
    product.apply(&x, &mut y1);
    a.apply(&x, &mut y2);
    
    assert_vec_close(&y1, &y2, EPSILON, "I*A = A");
}

#[test]
fn test_scale_by_zero() {
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(5.0, 0.0), c64::new(7.0, 0.0)]);
    let scaled = a.scale(0.0);
    
    let x = test_vector(2);
    let mut y = vec![c64::zero(); 2];
    
    scaled.apply(&x, &mut y);
    
    for i in 0..2 {
        assert!(y[i].norm() < EPSILON, "Scale by zero should give zero");
    }
}

#[test]
fn test_scale_by_one() {
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(5.0, 0.0), c64::new(7.0, 0.0)]);
    let scaled = a.scale(1.0);
    
    let x = test_vector(2);
    let mut y1 = vec![c64::zero(); 2];
    let mut y2 = vec![c64::zero(); 2];
    
    scaled.apply(&x, &mut y1);
    a.apply(&x, &mut y2);
    
    assert_vec_close(&y1, &y2, EPSILON, "Scale by one should not change operator");
}

// ============================================================================
// Mixed Type Tests (Dense and Sparse base operators)
// ============================================================================

#[test]
fn test_composed_from_dense_operator() {
    let dense = arr2(&[
        [c64::new(1.0, 0.0), c64::new(0.0, 0.0)],
        [c64::new(0.0, 0.0), c64::new(2.0, 0.0)],
    ]);
    
    let dense_op = dense;
    let composed = ComposedOperator::from_operator(dense_op);
    
    let x = vec![c64::one(); 2];
    let mut y = vec![c64::zero(); 2];
    
    composed.apply(&x, &mut y);
    
    let expected = vec![c64::new(1.0, 0.0), c64::new(2.0, 0.0)];
    assert_vec_close(&y, &expected, EPSILON, "From dense operator");
}

#[test]
fn test_composed_from_sparse_operator() {
    let sparse: CsMat<c64> = MatrixOps::from_diag(&[c64::new(3.0, 0.0), c64::new(4.0, 0.0)]);
    let composed = ComposedOperator::from_operator(sparse);
    
    let x = vec![c64::one(); 2];
    let mut y = vec![c64::zero(); 2];
    
    composed.apply(&x, &mut y);
    
    let expected = vec![c64::new(3.0, 0.0), c64::new(4.0, 0.0)];
    assert_vec_close(&y, &expected, EPSILON, "From sparse operator");
}
