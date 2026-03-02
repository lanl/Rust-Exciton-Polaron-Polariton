//! Test that KronSum optimization produces the same results as the unoptimized version
//! by building the same operator twice: once allowing KronSum, once forcing no optimization

use ndarray::{Array2};
use ndarray_linalg::c64;
use num_traits::{Zero, One};
use rand::Rng;

use light_matter::utils::{
    ComposedOperator, MatrixOps
};

fn main() {
    println!("\n=== Testing KronSum Correctness ===\n");

    let n_exciton = 5;
    let n_photon = 4; // Including ground state
    let n_modes = n_photon - 1;
    
    println!("System: {} exciton states, {} photon states ({} modes)", n_exciton, n_photon, n_modes);
    println!("Total dimension: {}\n", n_exciton * n_photon);

    // Create random exciton matrices
    let mut rng = rand::rng();
    let mut exciton_matrices: Vec<Array2<c64>> = Vec::new();
    
    for i in 0..n_modes {
        let mut mat = Array2::zeros((n_exciton, n_exciton));
        for row in 0..n_exciton {
            for col in 0..n_exciton {
                mat[[row, col]] = c64::new(
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0)
                );
            }
        }
        exciton_matrices.push(mat);
        println!("Exciton matrix {} max norm: {:.3e}", i, 
            exciton_matrices[i].iter().map(|v| v.norm()).fold(0.0f64, f64::max));
    }
    
    // Create photon operators (single excitation: |0⟩⟨i| and |i⟩⟨0|)
    let mut a_ops: Vec<Array2<c64>> = Vec::new();
    let mut adag_ops: Vec<Array2<c64>> = Vec::new();
    
    for mode in 0..n_modes {
        let mut a = Array2::zeros((n_photon, n_photon));
        a[[0, mode + 1]] = c64::one(); // |0⟩⟨mode+1| (annihilation)
        a_ops.push(a.clone());
        
        let mut adag = Array2::zeros((n_photon, n_photon));
        adag[[mode + 1, 0]] = c64::one(); // |mode+1⟩⟨0| (creation)
        adag_ops.push(adag);
    }
    
    // Build operator WITHOUT KronSum (manual expansion)
    println!("\nBuilding operator WITHOUT KronSum optimization...");
    let mut terms_no_opt: Vec<ComposedOperator<c64>> = Vec::new();
    
    for mode in 0..n_modes {
        let ex_op = ComposedOperator::from_operator(exciton_matrices[mode].clone());
        let a_op = ComposedOperator::from_operator(a_ops[mode].clone());
        let adag_op = ComposedOperator::from_operator(adag_ops[mode].clone());
        
        // Add both M ⊗ a and M ⊗ a†
        terms_no_opt.push(ex_op.kron(&a_op));
        terms_no_opt.push(ex_op.kron(&adag_op));
    }
    
    // Use regular Sum (not the optimizing version)
    let op_no_opt = ComposedOperator::Sum(terms_no_opt.into_iter().map(Box::new).collect());
    println!("  Created Sum with {} terms", 
        if let ComposedOperator::Sum(ref terms) = op_no_opt { terms.len() } else { 0 });
    
    // Build operator WITH KronSum (using the optimizing sum)
    println!("\nBuilding operator WITH KronSum optimization...");
    let mut terms_with_opt: Vec<ComposedOperator<c64>> = Vec::new();
    
    for mode in 0..n_modes {
        let ex_op = ComposedOperator::from_operator(exciton_matrices[mode].clone());
        let a_op = ComposedOperator::from_operator(a_ops[mode].clone());
        let adag_op = ComposedOperator::from_operator(adag_ops[mode].clone());
        
        terms_with_opt.push(ex_op.kron(&a_op));
        terms_with_opt.push(ex_op.kron(&adag_op));
    }
    
    // Use the optimizing sum function
    let op_with_opt = ComposedOperator::sum(terms_with_opt);
    
    // Check if KronSum was applied
    fn check_structure(op: &ComposedOperator<c64>, depth: usize) -> bool {
        let indent = "  ".repeat(depth);
        match op {
            ComposedOperator::KronSum { transition_matrices, .. } => {
                println!("{}KronSum with {} transitions", indent, transition_matrices.len());
                true
            }
            ComposedOperator::Sum(terms) => {
                println!("{}Sum with {} terms", indent, terms.len());
                terms.iter().any(|t| check_structure(t, depth + 1))
            }
            ComposedOperator::Add(left, right) => {
                println!("{}Add", indent);
                check_structure(left, depth + 1) || check_structure(right, depth + 1)
            }
            _ => false,
        }
    }
    
    let has_kronsum = check_structure(&op_with_opt, 1);
    
    if has_kronsum {
        println!("  ✓ KronSum optimization was applied\n");
    } else {
        println!("  ✗ KronSum optimization was NOT applied\n");
    }
    
    // Test with random vector
    println!("Testing operator application...");
    let dim = n_exciton * n_photon;
    let x: Vec<c64> = (0..dim).map(|i| {
        c64::new(
            (i as f64 * 0.1).cos(),
            (i as f64 * 0.1).sin()
        )
    }).collect();
    
    let mut y_no_opt = vec![c64::zero(); dim];
    let mut y_with_opt = vec![c64::zero(); dim];
    
    op_no_opt.apply(&x, &mut y_no_opt);
    op_with_opt.apply(&x, &mut y_with_opt);
    
    // Compare results
    let mut max_diff = 0.0f64;
    let mut max_diff_idx = 0;
    let mut total_diff = 0.0f64;
    
    for i in 0..dim {
        let diff = (y_no_opt[i] - y_with_opt[i]).norm();
        total_diff += diff;
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    
    let avg_diff = total_diff / dim as f64;
    let y_norm: f64 = y_no_opt.iter().map(|v| v.norm()).sum();
    
    println!("\nResults:");
    println!("  Output norm (no opt): {:.6e}", y_norm);
    println!("  Max difference: {:.6e} at index {}", max_diff, max_diff_idx);
    println!("  Average difference: {:.6e}", avg_diff);
    println!("  Relative error: {:.6e}", max_diff / (y_norm / dim as f64));
    
    if max_diff < 1e-10 {
        println!("\n  ✓✓✓ PASS: Results match perfectly!");
    } else {
        println!("\n  ✗✗✗ FAIL: Results differ significantly!");
        println!("\n  Details at max diff index {}:", max_diff_idx);
        println!("    No opt:   {:?}", y_no_opt[max_diff_idx]);
        println!("    With opt: {:?}", y_with_opt[max_diff_idx]);
        println!("    Input:    {:?}", x[max_diff_idx]);
    }
    
    println!("\nTest complete!");
}
