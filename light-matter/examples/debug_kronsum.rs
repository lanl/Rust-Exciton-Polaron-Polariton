//! Debug script to inspect KronSum transition matrices

use ndarray_linalg::c64;
use light_matter::utils::{
    Vector3D, Polarization, BosonMode, 
    ComposedOperator
};
use light_matter::hamiltonians::{
    LMHamiltonian, ExcitonicHamiltonian, BosonicHamiltonian, 
    SingExBosons
};
use exciton_2d::ExcitonConfig;

fn main() {
    env_logger::init();
    println!("\n=== Debug KronSum Transition Matrices ===\n");

    let n_photon_modes = 2; // Small system
    let n_exciton_states = 3;
    let n_rel = 9;
    let lm_coupling = 0.01;

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

    let q_values: Vec<Vector3D> = (0..n_photon_modes).map(|i| {
        let q = 0.1 + 0.05 * i as f64;
        Vector3D::new(q, 0.0, 0.0)
    }).collect();

    use exciton_2d::Exciton2DGridSystem;
    let exciton_grid = Exciton2DGridSystem::from_config(e_config.clone());
    let mut exciton: ExcitonicHamiltonian<ComposedOperator<c64>> = 
        ExcitonicHamiltonian::new(exciton_grid.full_ham, n_exciton_states, n_rel);
    exciton.compute_eigenstates();
    
    let photon_modes: Vec<BosonMode> = q_values.iter().map(|q| {
        BosonMode::new(*q, Polarization::TE, 2)
    }).collect();
    let photon: BosonicHamiltonian<ComposedOperator<c64>, SingExBosons<ComposedOperator<c64>>> = 
        BosonicHamiltonian::new(photon_modes);
    
    let mut lm_hamiltonian = LMHamiltonian::new(exciton, photon, lm_coupling);
    
    // Debug: Check matrix elements before building Hamiltonian
    println!("\nChecking matrix elements...");
    lm_hamiltonian.initialize(None);
    
    if let Some(p_c_sin_vec) = &lm_hamiltonian.p_c_sin {
        println!("p_c_sin matrices: {} modes", p_c_sin_vec.len());
        for (i, _) in p_c_sin_vec.iter().enumerate().take(2) {
            println!("  Mode {}: dimension {}", i, p_c_sin_vec[i].dimension());
        }
    }
    
    println!("p_cos matrices from matrix_elements:");
    for i in 0..2 {
        if let Some(p_cos_mat) = lm_hamiltonian.matrix_elements.p_cos_mat.get(&vec![i as i32]) {
            println!("  Mode {}: shape {:?}", i, p_cos_mat.dim());
            println!("    [0,0] = {:?}", p_cos_mat[[0,0]]);
            println!("    [0,1] = {:?}", p_cos_mat[[0,1]]);
            let max_val = p_cos_mat.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
            println!("    max norm = {:.2e}", max_val);
        }
    }
    
    println!("sin matrices from matrix_elements:");
    for i in 0..2 {
        if let Some(sin_mat) = lm_hamiltonian.matrix_elements.sin_mat.get(&vec![i as i32]) {
            println!("  Mode {}: shape {:?}", i, sin_mat.dim());
            println!("    [0,0] = {:?}", sin_mat[[0,0]]);
            println!("    [0,1] = {:?}", sin_mat[[0,1]]);
            let max_val = sin_mat.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
            println!("    max norm = {:.2e}", max_val);
            
            // Test: convert to ComposedOperator and apply
            if i == 0 {
                use light_matter::utils::MatrixOps;
                let sin_op = ComposedOperator::from_dense(sin_mat);
                let mut x = vec![c64::ZERO; 3];
                let mut y = vec![c64::ZERO; 3];
                x[0] = c64::new(1.0, 0.0);
                sin_op.apply(&x, &mut y);
                let y_nonzero = y.iter().any(|&v| v.norm() > 1e-10);
                println!("    Test: sin_op.apply produces non-zero output: {}", y_nonzero);
                if y_nonzero {
                    println!("      y[0] = {:?}, y[1] = {:?}", y[0], y[1]);
                }
            }
        }
    }
    
    let ham_composed = lm_hamiltonian.build_lm_hamiltonian(None);

    // Recursively find KronSum in the operator tree
    fn find_kronsum<'a>(op: &'a ComposedOperator<c64>) -> Option<&'a ComposedOperator<c64>> {
        match op {
            ComposedOperator::KronSum { .. } => Some(op),
            ComposedOperator::Add(left, right) => {
                find_kronsum(left).or_else(|| find_kronsum(right))
            }
            ComposedOperator::Sum(terms) => {
                terms.iter().find_map(|t| find_kronsum(t))
            }
            _ => None,
        }
    }
    
    // Extract and inspect the KronSum
    if let Some(ComposedOperator::KronSum { transition_matrices, exciton_dim, photon_dim }) = find_kronsum(&ham_composed) {
        println!("Found KronSum with {} transitions", transition_matrices.len());
        println!("Exciton dim: {}, Photon dim: {}\n", exciton_dim, photon_dim);
        
        // Print all transitions sorted
        let mut sorted_keys: Vec<_> = transition_matrices.keys().collect();
        sorted_keys.sort();
        
        for &key in sorted_keys.iter() {
            let matrix = &transition_matrices[key];
            println!("Transition ({} -> {})", key.0, key.1);
            println!("  Matrix shape: {:?}", matrix.dim());
            println!("  Matrix[0,0] = {:?}", matrix[[0, 0]]);
            println!("  Matrix[0,1] = {:?}", matrix[[0, 1]]);
            println!("  Matrix[1,0] = {:?}", matrix[[1, 0]]);
            
            // Check if matrix has imaginary components
            let has_imag = matrix.iter().any(|&v| v.im.abs() > 1e-10);
            let max_imag = matrix.iter().map(|v| v.im.abs()).fold(0.0f64, f64::max);
            println!("  Has imaginary: {}, max |Im| = {:.2e}\n", has_imag, max_imag);
        }
    } else {
        println!("No KronSum found in operator tree");
        // Print the structure
        fn print_structure(op: &ComposedOperator<c64>, depth: usize) {
            let indent = "  ".repeat(depth);
            match op {
                ComposedOperator::Add(left, right) => {
                    println!("{}Add:", indent);
                    print_structure(left, depth + 1);
                    print_structure(right, depth + 1);
                }
                ComposedOperator::Sum(terms) => {
                    println!("{}Sum({} terms):", indent, terms.len());
                    for t in terms.iter().take(3) {
                        print_structure(t, depth + 1);
                    }
                }
                ComposedOperator::Kron(left, right) => {
                    println!("{}Kron:", indent);
                    print_structure(left, depth + 1);
                    print_structure(right, depth + 1);
                }
                other => {
                    println!("{}{:?}", indent, std::mem::discriminant(other));
                }
            }
        }
        println!("\nOperator structure:");
        print_structure(&ham_composed, 0);
    }
}