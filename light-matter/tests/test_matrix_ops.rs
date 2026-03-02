/// Comprehensive tests for MatrixOps trait implementations
/// 
/// This test suite verifies that MatrixOps works correctly for:
/// - Dense matrices (Array2<f64> and Array2<c64>)
/// - Sparse matrices (CsMat<f64> and CsMat<c64>)
/// - Composed operators (ComposedOperator<c64>)

use ndarray::{Array2, arr1, arr2, linalg::Dot};
use ndarray_linalg::c64;
use num_traits::{One, Zero};
use sprs::{CsMat, TriMat};
use light_matter::utils::{MatrixOps, ComposedOperator};

const EPSILON: f64 = 1e-10;

// ============================================================================
// Helper Functions
// ============================================================================

// /// Create a simple 3x3 test matrix
// fn test_matrix_dense() -> Array2<f64> {
//     arr2(&[
//         [1.0, 2.0, 3.0],
//         [4.0, 5.0, 6.0],
//         [7.0, 8.0, 9.0],
//     ])
// }

// /// Create a simple 3x3 diagonal sparse matrix
// fn test_matrix_sparse() -> CsMat<f64> {
//     let mut trimat = TriMat::new((3, 3));
//     trimat.add_triplet(0, 0, 1.0);
//     trimat.add_triplet(1, 1, 2.0);
//     trimat.add_triplet(2, 2, 3.0);
//     trimat.to_csr()
// }

// /// Create a simple 3x3 complex matrix
// fn test_matrix_complex() -> Array2<c64> {
//     arr2(&[
//         [c64::new(1.0, 0.0), c64::new(2.0, 1.0), c64::new(3.0, 0.0)],
//         [c64::new(2.0, -1.0), c64::new(5.0, 0.0), c64::new(6.0, 2.0)],
//         [c64::new(3.0, 0.0), c64::new(6.0, -2.0), c64::new(9.0, 0.0)],
//     ])
// }

/// Compare two dense matrices for approximate equality
fn assert_array2_close(a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
    assert_eq!(a.shape(), b.shape(), "Matrix shapes don't match");
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let diff = (a[[i, j]] - b[[i, j]]).abs();
            assert!(
                diff < tol,
                "Matrices differ at ({}, {}): {} vs {} (diff: {})",
                i, j, a[[i, j]], b[[i, j]], diff
            );
        }
    }
}

// /// Compare two complex matrices for approximate equality
// fn assert_array2_complex_close(a: &Array2<c64>, b: &Array2<c64>, tol: f64) {
//     assert_eq!(a.shape(), b.shape(), "Matrix shapes don't match");
//     for i in 0..a.nrows() {
//         for j in 0..a.ncols() {
//             let diff = (a[[i, j]] - b[[i, j]]).norm();
//             assert!(
//                 diff < tol,
//                 "Matrices differ at ({}, {}): {} vs {} (diff: {})",
//                 i, j, a[[i, j]], b[[i, j]], diff
//             );
//         }
//     }
// }

// ============================================================================
// Dense Matrix Tests (f64)
// ============================================================================

#[test]
fn test_dense_eye() {
    let eye: Array2<f64> = MatrixOps::eye(3);
    let expected = Array2::eye(3);
    assert_array2_close(&eye, &expected, EPSILON);
}

#[test]
fn test_dense_zeros() {
    let zeros: Array2<f64> = MatrixOps::zeros((3, 3));
    assert_eq!(zeros.shape(), &[3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(zeros[[i, j]], 0.0);
        }
    }
}

#[test]
fn test_dense_from_diag() {
    let diag = vec![1.0, 2.0, 3.0];
    let mat: Array2<f64> = MatrixOps::from_diag(&diag);
    
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                assert_eq!(mat[[i, j]], diag[i]);
            } else {
                assert_eq!(mat[[i, j]], 0.0);
            }
        }
    }
}

#[test]
fn test_dense_kron() {
    let a: Array2<f64> = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b: Array2<f64> = arr2(&[[0.0, 5.0], [6.0, 7.0]]);
    
    let result = a.kron(&b);
    
    // Verify shape
    assert_eq!(result.shape(), &[4, 4]);
    
    // Verify some elements (Kronecker product properties)
    assert_eq!(result[[0, 0]], a[[0, 0]] * b[[0, 0]]);
    assert_eq!(result[[0, 1]], a[[0, 0]] * b[[0, 1]]);
    assert_eq!(result[[2, 2]], a[[1, 0]] * b[[0, 0]]);
}

#[test]
fn test_dense_add() {
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
    
    let result = a.add(&b);
    let expected = arr2(&[[6.0, 8.0], [10.0, 12.0]]);
    
    assert_array2_close(&result, &expected, EPSILON);
}

#[test]
fn test_dense_dot() {
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
    
    let result = MatrixOps::dot(&a,&b);
    let expected = arr2(&[[19.0, 22.0], [43.0, 50.0]]);
    
    assert_array2_close(&result, &expected, EPSILON);
}

#[test]
fn test_dense_transpose() {
    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let result = a.transpose();
    let expected = arr2(&[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    
    assert_array2_close(&result, &expected, EPSILON);
}

#[test]
fn test_dense_scale() {
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let result = a.scale(2.0);
    let expected = arr2(&[[2.0, 4.0], [6.0, 8.0]]);
    
    assert_array2_close(&result, &expected, EPSILON);
}

#[test]
fn test_dense_from_trimat() {
    let mut trimat = TriMat::new((3, 3));
    trimat.add_triplet(0, 0, 1.0);
    trimat.add_triplet(0, 1, 2.0);
    trimat.add_triplet(1, 1, 3.0);
    trimat.add_triplet(2, 2, 4.0);
    
    let dense: Array2<f64> = MatrixOps::from_trimat(&trimat);
    
    assert_eq!(dense[[0, 0]], 1.0);
    assert_eq!(dense[[0, 1]], 2.0);
    assert_eq!(dense[[1, 1]], 3.0);
    assert_eq!(dense[[2, 2]], 4.0);
    assert_eq!(dense[[1, 0]], 0.0);
}

#[test]
fn test_dense_add_trimat() {
    let mut mat: Array2<f64> = Array2::zeros((3, 3));
    
    let mut trimat = TriMat::new((3, 3));
    trimat.add_triplet(0, 0, 1.0);
    trimat.add_triplet(1, 1, 2.0);
    trimat.add_triplet(2, 2, 3.0);
    
    mat.add_trimat(&trimat);
    
    assert_eq!(mat[[0, 0]], 1.0);
    assert_eq!(mat[[1, 1]], 2.0);
    assert_eq!(mat[[2, 2]], 3.0);
}

// ============================================================================
// Sparse Matrix Tests (f64)
// ============================================================================

#[test]
fn test_sparse_eye() {
    let eye: CsMat<f64> = MatrixOps::eye(3);
    
    // Check diagonal elements
    for (i, &val) in eye.diag().iter() {
        assert_eq!(val, 1.0);
        assert!(i < 3);
    }
    
    // Check total number of non-zeros
    assert_eq!(eye.nnz(), 3);
}

#[test]
fn test_sparse_zeros() {
    let zeros: CsMat<f64> = MatrixOps::zeros((3, 3));
    assert_eq!(zeros.shape(), (3, 3));
    assert_eq!(zeros.nnz(), 0);
}

#[test]
fn test_sparse_from_diag() {
    let diag = vec![1.0, 2.0, 3.0];
    let mat: CsMat<f64> = MatrixOps::from_diag(&diag);
    
    assert_eq!(mat.nnz(), 3);
    
    let diag_vals: Vec<f64> = mat.diag().iter().map(|(_, &v)| v).collect();
    assert_eq!(diag_vals, diag);
}

#[test]
fn test_sparse_kron() {
    let a: CsMat<f64> = MatrixOps::from_diag(&[1.0, 2.0]);
    let b: CsMat<f64> = MatrixOps::from_diag(&[3.0, 4.0]);
    
    let result = a.kron(&b);
    
    assert_eq!(result.shape(), (4, 4));
    // Diagonal of kron product of diagonals
    let diag_vals: Vec<f64> = result.diag().iter().map(|(_, &v)| v).collect();
    assert_eq!(diag_vals, vec![3.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_sparse_add() {
    let a: CsMat<f64> = MatrixOps::from_diag(&[1.0, 2.0, 3.0]);
    let b: CsMat<f64> = MatrixOps::from_diag(&[4.0, 5.0, 6.0]);
    
    let result = a.add(&b);
    
    let diag_vals: Vec<f64> = result.diag().iter().map(|(_, &v)| v).collect();
    assert_eq!(diag_vals, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_sparse_dot() {
    let a: CsMat<f64> = MatrixOps::from_diag(&[1.0, 2.0, 3.0]);
    let b: CsMat<f64> = MatrixOps::from_diag(&[4.0, 5.0, 6.0]);
    
    let result = MatrixOps::dot(&a,&b);
    
    // Diagonal matrix multiplication
    let diag_vals: Vec<f64> = result.diag().iter().map(|(_, &v)| v).collect();
    assert_eq!(diag_vals, vec![4.0, 10.0, 18.0]);
}

#[test]
fn test_sparse_transpose() {
    let mut trimat = TriMat::new((3, 2));
    trimat.add_triplet(0, 0, 1.0);
    trimat.add_triplet(0, 1, 2.0);
    trimat.add_triplet(1, 0, 3.0);
    
    let mat: CsMat<f64> = MatrixOps::from_trimat(&trimat);
    let result = mat.transpose();
    
    assert_eq!(result.shape(), (2, 3));
    
    // Convert to dense for easier verification
    let dense = result.to_dense();
    assert_eq!(dense[[0, 0]], 1.0);
    assert_eq!(dense[[1, 0]], 2.0);
    assert_eq!(dense[[0, 1]], 3.0);
}

#[test]
fn test_sparse_scale() {
    let mat: CsMat<f64> = MatrixOps::from_diag(&[1.0, 2.0, 3.0]);
    let result = mat.scale(2.0);
    
    let diag_vals: Vec<f64> = result.diag().iter().map(|(_, &v)| v).collect();
    assert_eq!(diag_vals, vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_sparse_from_dense() {
    let dense = arr2(&[[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]);
    let sparse: CsMat<f64> = MatrixOps::from_dense(&dense);
    
    // Should have 5 non-zero elements
    assert_eq!(sparse.nnz(), 5);
    
    // Convert back to dense for verification
    let result: Array2<f64> = sparse.to_dense();
    assert_array2_close(&result, &dense, EPSILON);
}

// ============================================================================
// Complex Matrix Tests
// ============================================================================

#[test]
fn test_complex_dense_eye() {
    let eye: Array2<c64> = MatrixOps::eye(3);
    
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                assert!((eye[[i, j]] - c64::one()).norm() < EPSILON);
            } else {
                assert!(eye[[i, j]].norm() < EPSILON);
            }
        }
    }
}

#[test]
fn test_complex_dense_kron() {
    let a: Array2<c64> = arr2(&[
        [c64::new(1.0, 0.0), c64::new(0.0, 1.0)],
        [c64::new(0.0, -1.0), c64::new(1.0, 0.0)],
    ]);
    let b: Array2<c64> = arr2(&[
        [c64::new(2.0, 0.0), c64::new(0.0, 0.0)],
        [c64::new(0.0, 0.0), c64::new(2.0, 0.0)],
    ]);
    
    let result = a.kron(&b);
    
    assert_eq!(result.shape(), &[4, 4]);
    
    // Check first element
    assert!((result[[0, 0]] - c64::new(2.0, 0.0)).norm() < EPSILON);
}

#[test]
fn test_complex_sparse_operations() {
    let mut trimat = TriMat::new((2, 2));
    trimat.add_triplet(0, 0, c64::new(1.0, 1.0));
    trimat.add_triplet(1, 1, c64::new(2.0, -1.0));
    
    let mat: CsMat<c64> = trimat.to_csr();
    
    // Test that we can create and manipulate complex sparse matrices
    assert_eq!(mat.nnz(), 2);
    
    let eye: CsMat<c64> = MatrixOps::eye(2);
    let result = mat.add(&eye);
    
    // Should have 2 non-zero elements
    assert_eq!(result.nnz(), 2);
}

// ============================================================================
// ComposedOperator Tests
// ============================================================================

#[test]
fn test_composed_identity() {
    let id: ComposedOperator<c64> = MatrixOps::eye(3);
    
    assert_eq!(id.dimension(), 3);
    
    let x = vec![c64::new(1.0, 0.0), c64::new(2.0, 0.0), c64::new(3.0, 0.0)];
    let mut y = vec![c64::zero(); 3];
    
    id.apply(&x, &mut y);
    
    for i in 0..3 {
        assert!((y[i] - x[i]).norm() < EPSILON);
    }
}

#[test]
fn test_composed_diagonal() {
    let diag = vec![c64::new(1.0, 0.0), c64::new(2.0, 0.0), c64::new(3.0, 0.0)];
    let op: ComposedOperator<c64> = MatrixOps::from_diag(&diag);
    
    let x = vec![c64::new(1.0, 0.0), c64::new(1.0, 0.0), c64::new(1.0, 0.0)];
    let mut y = vec![c64::zero(); 3];
    
    op.apply(&x, &mut y);
    
    for i in 0..3 {
        assert!((y[i] - diag[i]).norm() < EPSILON);
    }
}

#[test]
fn test_composed_add() {
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(1.0, 0.0), c64::new(2.0, 0.0)]);
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(3.0, 0.0), c64::new(4.0, 0.0)]);
    
    let result = a.add(&b);
    
    let x = vec![c64::new(1.0, 0.0), c64::new(1.0, 0.0)];
    let mut y = vec![c64::zero(); 2];
    
    result.apply(&x, &mut y);
    
    assert!((y[0] - c64::new(4.0, 0.0)).norm() < EPSILON);
    assert!((y[1] - c64::new(6.0, 0.0)).norm() < EPSILON);
}

#[test]
fn test_composed_kron() {
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(1.0, 0.0), c64::new(2.0, 0.0)]);
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(3.0, 0.0), c64::new(4.0, 0.0)]);
    
    let result = a.kron(&b);
    
    assert_eq!(result.dimension(), 4);
    
    // Kronecker product of diagonal matrices is diagonal
    let diag = result.get_diagonal();
    
    assert!((diag[0] - c64::new(3.0, 0.0)).norm() < EPSILON);
    assert!((diag[1] - c64::new(4.0, 0.0)).norm() < EPSILON);
    assert!((diag[2] - c64::new(6.0, 0.0)).norm() < EPSILON);
    assert!((diag[3] - c64::new(8.0, 0.0)).norm() < EPSILON);
}

#[test]
fn test_composed_scale() {
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(1.0, 0.0), c64::new(2.0, 0.0)]);
    let result = a.scale(2.0);
    
    let x = vec![c64::new(1.0, 0.0), c64::new(1.0, 0.0)];
    let mut y = vec![c64::zero(); 2];
    
    result.apply(&x, &mut y);
    
    assert!((y[0] - c64::new(2.0, 0.0)).norm() < EPSILON);
    assert!((y[1] - c64::new(4.0, 0.0)).norm() < EPSILON);
}

#[test]
fn test_composed_dot() {
    // Create two diagonal operators
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(2.0, 0.0), c64::new(3.0, 0.0)]);
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(4.0, 0.0), c64::new(5.0, 0.0)]);
    
    let result = a.dot(&b);
    
    let x = vec![c64::new(1.0, 0.0), c64::new(1.0, 0.0)];
    let mut y = vec![c64::zero(); 2];
    
    result.apply(&x, &mut y);
    
    // (2*4)*1 = 8, (3*5)*1 = 15
    assert!((y[0] - c64::new(8.0, 0.0)).norm() < EPSILON);
    assert!((y[1] - c64::new(15.0, 0.0)).norm() < EPSILON);
}

#[test]
fn test_composed_transpose() {
    // Create a non-symmetric matrix as a base operator
    let dense = arr2(&[
        [c64::new(1.0, 0.0), c64::new(2.0, 0.0)],
        [c64::new(3.0, 0.0), c64::new(4.0, 0.0)],
    ]);
    let op = ComposedOperator::from_operator(dense);
    
    let transposed = op.transpose();
    
    // Apply to test vector
    let x = vec![c64::new(1.0, 0.0), c64::new(1.0, 0.0)];
    let mut y = vec![c64::zero(); 2];
    
    transposed.apply(&x, &mut y);
    
    // Transpose of [[1,2],[3,4]] applied to [1,1] = [4, 6]
    assert!((y[0] - c64::new(4.0, 0.0)).norm() < EPSILON);
    assert!((y[1] - c64::new(6.0, 0.0)).norm() < EPSILON);
}

#[test]
fn test_composed_from_operator() {
    let dense = arr2(&[
        [c64::new(1.0, 0.0), c64::new(0.0, 0.0)],
        [c64::new(0.0, 0.0), c64::new(2.0, 0.0)],
    ]);
    let dense_op = dense;
    let composed = ComposedOperator::from_operator(dense_op);
    
    assert_eq!(composed.dimension(), 2);
    
    let x = vec![c64::new(1.0, 0.0), c64::new(1.0, 0.0)];
    let mut y = vec![c64::zero(); 2];
    
    composed.apply(&x, &mut y);
    
    assert!((y[0] - c64::new(1.0, 0.0)).norm() < EPSILON);
    assert!((y[1] - c64::new(2.0, 0.0)).norm() < EPSILON);
}

// ============================================================================
// Integration Tests: Dense <-> Sparse Consistency
// ============================================================================

#[test]
fn test_dense_sparse_equivalence() {
    // Create identical matrices in dense and sparse formats
    let dense = arr2(&[[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]);
    let sparse: CsMat<f64> = MatrixOps::from_dense(&dense);
    
    // Test that operations give the same results
    let v = arr1(&[1.0, 2.0, 3.0]);
    
    let dense_result = Dot::dot(&dense,&v);
    
    // Note: sparse matrix-vector multiply accumulates, so we need to handle this differently
    let mut sparse_vec = vec![0.0; 3];
    sprs::prod::mul_acc_mat_vec_csr(sparse.view(), v.as_slice().unwrap(), &mut sparse_vec);
    
    for i in 0..3 {
        assert!((dense_result[i] - sparse_vec[i]).abs() < EPSILON);
    }
}

#[test]
fn test_kron_dense_sparse_equivalence() {
    let diag_vals = vec![1.0, 2.0];
    let dense: Array2<f64> = MatrixOps::from_diag(&diag_vals);
    let sparse: CsMat<f64> = MatrixOps::from_diag(&diag_vals);
    
    let dense_kron = dense.kron(&dense);
    let sparse_kron = sparse.kron(&sparse);
    
    // Convert sparse to dense for comparison
    let sparse_kron_dense = sparse_kron.to_dense();
    
    assert_array2_close(&dense_kron, &sparse_kron_dense, EPSILON);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_empty_from_diag() {
    let empty_diag: Vec<f64> = vec![];
    let mat: Array2<f64> = MatrixOps::from_diag(&empty_diag);
    
    assert_eq!(mat.shape(), &[0, 0]);
}

#[test]
fn test_single_element() {
    let diag = vec![42.0];
    let mat: Array2<f64> = MatrixOps::from_diag(&diag);
    
    assert_eq!(mat.shape(), &[1, 1]);
    assert_eq!(mat[[0, 0]], 42.0);
}

#[test]
fn test_kron_with_identity() {
    let a: Array2<f64> = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let id: Array2<f64> = MatrixOps::eye(2);
    
    // A ⊗ I should preserve A's structure
    let result = a.kron(&id);
    
    // Verify shape
    assert_eq!(result.shape(), &[4, 4]);
    
    // Check that A's elements are repeated along diagonal blocks
    assert_eq!(result[[0, 0]], 1.0);
    assert_eq!(result[[1, 1]], 1.0);
    assert_eq!(result[[2, 2]], 4.0);
    assert_eq!(result[[3, 3]], 4.0);
}

#[test]
fn test_composed_double_transpose() {
    let dense = arr2(&[
        [c64::new(1.0, 0.0), c64::new(2.0, 0.0)],
        [c64::new(3.0, 0.0), c64::new(4.0, 0.0)],
    ]);
    let op = ComposedOperator::from_operator(dense.clone());
    
    // (A^T)^T = A
    let double_transposed = op.transpose().transpose();
    
    let x = vec![c64::new(1.0, 0.0), c64::new(2.0, 0.0)];
    let mut y1 = vec![c64::zero(); 2];
    let mut y2 = vec![c64::zero(); 2];
    
    op.apply(&x, &mut y1);
    double_transposed.apply(&x, &mut y2);
    
    for i in 0..2 {
        assert!((y1[i] - y2[i]).norm() < EPSILON);
    }
}

// ============================================================================
// Performance/Stress Tests
// ============================================================================

#[test]
fn test_large_sparse_operations() {
    let n = 1000;
    let diag: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
    
    let mat: CsMat<f64> = MatrixOps::from_diag(&diag);
    
    assert_eq!(mat.nnz(), n);
    assert_eq!(mat.shape(), (n, n));
    
    // Test that operations don't panic
    let _ = mat.transpose();
    let _ = mat.scale(2.0);
}

#[test]
fn test_nested_composed_operations() {
    // Test deeply nested operations: ((A + B) ⊗ C) * scalar
    let a: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(1.0, 0.0)]);
    let b: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(2.0, 0.0)]);
    let c: ComposedOperator<c64> = MatrixOps::from_diag(&[c64::new(3.0, 0.0), c64::new(4.0, 0.0)]);
    
    let sum = a.add(&b);
    let kron_result = sum.kron(&c);
    let final_result = kron_result.scale(2.0);
    
    assert_eq!(final_result.dimension(), 2);
    
    let diag = final_result.get_diagonal();
    
    // (1+2) ⊗ [3,4] * 2 = 3 ⊗ [3,4] * 2 = [9, 12] * 2 = [18, 24]
    assert!((diag[0] - c64::new(18.0, 0.0)).norm() < EPSILON);
    assert!((diag[1] - c64::new(24.0, 0.0)).norm() < EPSILON);
}
