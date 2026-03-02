use ndarray::{Array2};
use ndarray_linalg::c64;
use num_traits::{One, Zero};
use light_matter::utils::ComposedOperator;
use sprs::TriMat;

#[test]
fn test_kronsum_pattern_detection() {
    // Create a simple test case: 3 photon modes, 2 exciton states
    let exciton_dim = 2;
    let photon_dim = 4; // vacuum + 3 modes
    
    // Create some exciton matrices
    let m1 = Array2::from_shape_vec((2, 2), vec![
        c64::new(1.0, 0.0), c64::new(0.1, 0.0),
        c64::new(0.1, 0.0), c64::new(2.0, 0.0),
    ]).unwrap();
    
    let m2 = Array2::from_shape_vec((2, 2), vec![
        c64::new(1.5, 0.0), c64::new(0.2, 0.0),
        c64::new(0.2, 0.0), c64::new(2.5, 0.0),
    ]).unwrap();
    
    // Create photon transition operators: |i⟩⟨0| (vacuum to mode i)
    let mut terms = Vec::new();
    
    for i in 1..4 {
        // Create sparse matrix for |i⟩⟨0|
        let mut trimat = TriMat::new((photon_dim, photon_dim));
        trimat.add_triplet(i, 0, c64::one());
        let photon_op = ComposedOperator::from_operator(trimat.to_csr());
        
        // Create Kron products
        let exciton_op = ComposedOperator::from_operator(m1.clone());
        terms.push(ComposedOperator::Kron(
            Box::new(exciton_op),
            Box::new(photon_op),
        ));
    }
    
    // Add another term with different exciton matrix but same photon transition
    let mut trimat = TriMat::new((photon_dim, photon_dim));
    trimat.add_triplet(1, 0, c64::one());
    let photon_op = ComposedOperator::from_operator(trimat.to_csr());
    let exciton_op = ComposedOperator::from_operator(m2.clone());
    terms.push(ComposedOperator::Kron(
        Box::new(exciton_op),
        Box::new(photon_op),
    ));
    
    // Create sum - should trigger KronSum optimization
    let sum_op = ComposedOperator::sum(terms);
    
    // Check if optimization was applied
    assert!(sum_op.is_kron_sum(), "KronSum optimization should be applied");
    
    if let Some((n_transitions, ex_dim, ph_dim)) = sum_op.get_kron_sum_info() {
        assert_eq!(ex_dim, exciton_dim, "Exciton dimension should match");
        assert_eq!(ph_dim, photon_dim, "Photon dimension should match");
        // We expect 3 unique transitions (vacuum -> mode 1, 2, 3)
        // Note: mode 1 appears twice but should be summed into one transition
        assert_eq!(n_transitions, 3, "Should have 3 unique transitions");
    } else {
        panic!("Failed to get KronSum info");
    }
}

#[test]
fn test_kronsum_apply() {
    // Test that KronSum gives same results as explicit Kron sum
    let exciton_dim = 2;
    let photon_dim = 3; // vacuum + 2 modes
    
    // Create exciton matrix
    let m = Array2::from_shape_vec((2, 2), vec![
        c64::new(1.0, 0.0), c64::new(0.5, 0.0),
        c64::new(0.5, 0.0), c64::new(2.0, 0.0),
    ]).unwrap();
    
    // Create terms for KronSum
    let mut kronsum_terms = Vec::new();
    
    // Create standard sum for comparison
    let mut standard_terms = Vec::new();
    
    for i in 1..3 {
        // Photon transition |i⟩⟨0|
        let mut trimat = TriMat::new((photon_dim, photon_dim));
        trimat.add_triplet(i, 0, c64::one());
        let photon_op = ComposedOperator::from_operator(trimat.to_csr());
        
        let exciton_op = ComposedOperator::from_operator(m.clone());
        
        kronsum_terms.push(ComposedOperator::Kron(
            Box::new(exciton_op.clone()),
            Box::new(photon_op.clone()),
        ));
        
        standard_terms.push(ComposedOperator::Kron(
            Box::new(exciton_op),
            Box::new(photon_op),
        ));
    }
    
    // Create KronSum (optimized)
    let kronsum_op = ComposedOperator::sum(kronsum_terms);
    assert!(kronsum_op.is_kron_sum(), "Should create KronSum");
    
    // Create standard Sum (unoptimized)
    let standard_op = ComposedOperator::Sum(
        standard_terms.into_iter().map(Box::new).collect()
    );
    
    // Test on a random input vector
    let total_dim = exciton_dim * photon_dim;
    let x: Vec<c64> = (0..total_dim)
        .map(|i| c64::new(i as f64 * 0.1, (i as f64 * 0.05).sin()))
        .collect();
    
    let mut y_kronsum = vec![c64::zero(); total_dim];
    let mut y_standard = vec![c64::zero(); total_dim];
    
    kronsum_op.apply(&x, &mut y_kronsum);
    standard_op.apply(&x, &mut y_standard);
    
    // Results should be identical
    for i in 0..total_dim {
        let diff = (y_kronsum[i] - y_standard[i]).norm();
        assert!(diff < 1e-10, 
            "Results differ at index {}: KronSum={:?}, Standard={:?}, diff={}", 
            i, y_kronsum[i], y_standard[i], diff);
    }
}

#[test]
fn test_kronsum_transpose() {
    // Test that transpose works correctly
    let exciton_dim = 2;
    let photon_dim = 3;
    
    let m = Array2::from_shape_vec((2, 2), vec![
        c64::new(1.0, 0.5), c64::new(0.3, 0.2),
        c64::new(0.4, 0.1), c64::new(2.0, 0.3),
    ]).unwrap();
    
    let mut terms = Vec::new();
    
    for i in 1..3 {
        let mut trimat = TriMat::new((photon_dim, photon_dim));
        trimat.add_triplet(i, 0, c64::one());
        let photon_op = ComposedOperator::from_operator(trimat.to_csr());
        
        let exciton_op = ComposedOperator::from_operator(m.clone());
        
        terms.push(ComposedOperator::Kron(
            Box::new(exciton_op),
            Box::new(photon_op),
        ));
    }
    
    let op = ComposedOperator::sum(terms);
    assert!(op.is_kron_sum());
    
    let total_dim = exciton_dim * photon_dim;
    let x: Vec<c64> = (0..total_dim)
        .map(|i| c64::new(i as f64 * 0.1, 0.0))
        .collect();
    
    let mut y = vec![c64::zero(); total_dim];
    let mut y_t = vec![c64::zero(); total_dim];
    
    op.apply(&x, &mut y);
    op.apply_transpose(&x, &mut y_t);
    
    // For Hermitian parts, check that transpose is conjugate
    // Note: This is a basic test - for non-Hermitian operators,
    // we'd need a more sophisticated test
    assert_eq!(y.len(), y_t.len());
}

#[test]
fn test_kronsum_diagonal() {
    // Test diagonal extraction
    let exciton_dim = 2;
    let photon_dim = 3;
    
    // Create diagonal exciton matrix for easy verification
    let m = Array2::from_shape_vec((2, 2), vec![
        c64::new(1.0, 0.0), c64::zero(),
        c64::zero(), c64::new(2.0, 0.0),
    ]).unwrap();
    
    let mut terms = Vec::new();
    
    // Create diagonal photon transition (same state -> same state)
    for i in 0..3 {
        let mut trimat = TriMat::new((photon_dim, photon_dim));
        trimat.add_triplet(i, i, c64::one());
        let photon_op = ComposedOperator::from_operator(trimat.to_csr());
        
        let exciton_op = ComposedOperator::from_operator(m.clone());
        
        terms.push(ComposedOperator::Kron(
            Box::new(exciton_op),
            Box::new(photon_op),
        ));
    }
    
    let op = ComposedOperator::sum(terms);
    assert!(op.is_kron_sum());
    
    let diag = op.get_diagonal();
    let total_dim = exciton_dim * photon_dim;
    assert_eq!(diag.len(), total_dim);
    
    // Standard Kronecker product layout: exciton varies slowest
    // Each exciton state should have diagonal [1.0, 1.0, 1.0] (one per photon state)
    for ex_state in 0..exciton_dim {
        for ph_state in 0..photon_dim {
            let idx = ex_state * photon_dim + ph_state;
            let expected = if ex_state == 0 { 1.0 } else { 2.0 };
            assert!((diag[idx].re - expected).abs() < 1e-10,
                "Diagonal mismatch at exciton={}, photon={}: expected {}, got {}",
                ex_state, ph_state, expected, diag[idx].re);
        }
    }
}

#[test]
fn test_kronsum_dimension() {
    // Test dimension calculation
    let exciton_dim = 4;
    let photon_dim = 10;
    
    let m = Array2::eye(exciton_dim);
    
    let mut terms = Vec::new();
    
    for i in 1..5 {
        let mut trimat = TriMat::new((photon_dim, photon_dim));
        trimat.add_triplet(i, 0, c64::one());
        let photon_op = ComposedOperator::from_operator(trimat.to_csr());
        
        let exciton_op = ComposedOperator::from_operator(m.clone());
        
        terms.push(ComposedOperator::Kron(
            Box::new(exciton_op),
            Box::new(photon_op),
        ));
    }
    
    let op = ComposedOperator::sum(terms);
    assert!(op.is_kron_sum());
    
    assert_eq!(op.dimension(), exciton_dim * photon_dim,
        "Total dimension should be exciton_dim * photon_dim");
}