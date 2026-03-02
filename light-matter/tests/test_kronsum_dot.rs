use ndarray_linalg::c64;
use num_traits::{One, Zero};
use light_matter::utils::{ComposedOperator, MatrixOps};
use light_matter::hamiltonians::{BosonicHamiltonian, SingExBosons};
use light_matter::utils::BosonMode;

#[test]
fn test_dot_composition() {
    // Create a simple SingExBosons system with 2 modes
    let modes = vec![
        BosonMode { eps: ndarray::arr1(&[1.0, 0.0]), q: ndarray::arr1(&[0.1, 0.0]), omega: 0.1, n_fock: 2 },
        BosonMode { eps: ndarray::arr1(&[1.0, 0.0]), q: ndarray::arr1(&[0.2, 0.0]), omega: 0.2, n_fock: 2 },
    ];
    
    let photon: BosonicHamiltonian<ComposedOperator<c64>, SingExBosons<ComposedOperator<c64>>> = 
        BosonicHamiltonian::new(modes);
    
    // Get operators
    let a_0 = photon.get_a(0);
    let a_1 = photon.get_a(1);
    
    println!("\nTesting Dot compositions:");
    
    // Test a_0.dot(&a_0.transpose()) - should be |0⟩⟨0|
    let a0_a0dag = a_0.dot(&a_0.transpose());
    println!("a_0 · a_0†: dim = {}", a0_a0dag.dimension());
    
    // Apply to vacuum state
    let x = vec![c64::one(), c64::zero(), c64::zero()];
    let mut y = vec![c64::zero(); 3];
    a0_a0dag.apply(&x, &mut y);
    println!("  Applied to |0⟩: {:?}", y);
    
    // Test a_0.dot(&a_1.transpose()) - should be zero
    let a0_a1dag = a_0.dot(&a_1.transpose());
    println!("\na_0 · a_1†: dim = {}", a0_a1dag.dimension());
    
    let mut y2 = vec![c64::zero(); 3];
    a0_a1dag.apply(&x, &mut y2);
    println!("  Applied to |0⟩: {:?}", y2);
    
    // Now test if these are recognized in a sum
    let terms = vec![a0_a0dag.clone(), a0_a1dag.clone()];
    let sum_op = ComposedOperator::sum(terms);
    
    println!("\nSum operator info:");
    if sum_op.is_kron_sum() {
        println!("  ✓ Recognized as KronSum");
        if let Some((n_trans, ex_dim, ph_dim)) = sum_op.get_kron_sum_info() {
            println!("    Transitions: {}, ex_dim: {}, ph_dim: {}", n_trans, ex_dim, ph_dim);
        }
    } else {
        println!("  ✗ Not recognized as KronSum");
    }
}
