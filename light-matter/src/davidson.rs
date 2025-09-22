use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::{Eigh, UPLO};
use std::fmt::Debug;

/// Trait for applying a linear operator to a vector without explicitly building the matrix
pub trait LinearOperator {
    /// Apply the operator to a vector: result = A * v
    fn apply(&self, v: &Array1<f64>, result: &mut Array1<f64>);
    
    /// Get the dimension of the operator
    fn dim(&self) -> usize;
    
    /// Get the diagonal elements of the operator (needed for preconditioning)
    fn diagonal(&self) -> Array1<f64>;
}

/// Davidson algorithm configuration parameters
pub struct DavidsonConfig {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Maximum subspace size before restart
    pub max_subspace_size: usize,
    /// Minimum subspace size after restart
    pub min_subspace_size: usize,
    /// Convergence threshold for residual norm
    pub tolerance: f64,
    /// Number of eigenvalues to compute
    pub n_eigenvalues: usize,
    /// Whether to use preconditioning
    pub use_preconditioning: bool,
}

impl Default for DavidsonConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            max_subspace_size: 20,
            min_subspace_size: 10,
            tolerance: 1e-8,
            n_eigenvalues: 1,
            use_preconditioning: true,
        }
    }
}

/// Result of the Davidson algorithm
#[derive(Debug)]
pub struct DavidsonResult {
    /// Computed eigenvalues
    pub eigenvalues: Array1<f64>,
    /// Computed eigenvectors
    pub eigenvectors: Array2<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Maximum residual norm
    pub max_residual: f64,
}

/// Davidson algorithm for diagonalizing real symmetric matrices
pub fn davidson<T: LinearOperator>(
    operator: &T,
    config: &DavidsonConfig,
    initial_guess: Option<Array2<f64>>,
) -> DavidsonResult {
    let n = operator.dim();
    let k = config.n_eigenvalues;
    
    // Validate parameters
    assert!(k < n, "Number of requested eigenvalues must be less than matrix dimension");
    assert!(config.min_subspace_size >= k, "Minimum subspace size must be at least the number of eigenvalues");
    assert!(config.max_subspace_size >= config.min_subspace_size, "Maximum subspace size must be at least the minimum subspace size");
    
    // Initialize the subspace with the initial guess or random vectors
    let mut v_basis = match initial_guess {
        Some(guess) => {
            assert_eq!(guess.shape()[0], n, "Initial guess must have correct dimension");
            assert!(guess.shape()[1] >= k, "Initial guess must have at least k columns");
            
            // Orthogonalize the initial guess using Gram-Schmidt
            let mut basis = Array2::zeros((n, guess.shape()[1]));
            for j in 0..guess.shape()[1] {
                let mut vj = guess.column(j).to_owned();
                
                // Orthogonalize against previous vectors
                for i in 0..j {
                    let vi = basis.column(i);
                    let projection = vi.dot(&vj);
                    vj = &vj - &(&vi * projection);
                }
                
                // Normalize
                let norm = vj.dot(&vj).sqrt();
                if norm > 1e-10 {
                    vj /= norm;
                    basis.column_mut(j).assign(&vj);
                } else {
                    // If linear dependence is detected, generate a random vector
                    let mut rng = rand::thread_rng();
                    let mut random_vec = Array1::zeros(n);
                    for i in 0..n {
                        random_vec[i] = rand::Rng::gen_range(&mut rng, -1.0..1.0);
                    }
                    
                    // Orthogonalize against previous vectors
                    for i in 0..j {
                        let vi = basis.column(i);
                        let projection = vi.dot(&random_vec);
                        random_vec = &random_vec - &(&vi * projection);
                    }
                    
                    // Normalize
                    let norm = random_vec.dot(&random_vec).sqrt();
                    random_vec /= norm;
                    basis.column_mut(j).assign(&random_vec);
                }
            }
            
            basis
        },
        None => {
            // Start with random orthogonal vectors
            use rand::Rng;
            let mut rng = rand::thread_rng();
            
            let mut basis = Array2::zeros((n, k));
            for j in 0..k {
                let mut vj = Array1::zeros(n);
                for i in 0..n {
                    vj[i] = rand::Rng::gen_range(&mut rng, -1.0..1.0);
                }
                
                // Orthogonalize against previous vectors
                for i in 0..j {
                    let vi = basis.column(i);
                    let projection = vi.dot(&vj);
                    vj = &vj - &(&vi * projection);
                }
                
                // Normalize
                let norm = vj.dot(&vj).sqrt();
                vj /= norm;
                basis.column_mut(j).assign(&vj);
            }
            
            basis
        }
    };
    
    // Get diagonal elements for preconditioning
    let diag = operator.diagonal();
    
    // Allocate work arrays
    let mut av = Array1::zeros(n);
    let mut subspace_matrix = Array2::zeros((v_basis.shape()[1], v_basis.shape()[1]));
    
    // Main Davidson iteration
    let mut iteration = 0;
    let mut converged = false;
    let mut max_residual = std::f64::MAX;
    
    while iteration < config.max_iter && !converged {
        let m = v_basis.shape()[1]; // Current subspace size
        
        // Compute the projected subspace matrix H_ij = <v_i|A|v_j>
        for j in 0..m {
            let vj = v_basis.column(j);
            operator.apply(&vj.to_owned(), &mut av);
            
            for i in 0..=j {
                let vi = v_basis.column(i);
                let h_ij = vi.dot(&av);
                subspace_matrix[(i, j)] = h_ij;
                subspace_matrix[(j, i)] = h_ij; // Symmetric matrix
            }
        }
        
        // Solve the eigenvalue problem in the subspace
        let subspace_eig = subspace_matrix.slice(s![0..m, 0..m]).eigh(UPLO::Upper).unwrap();
        let theta = subspace_eig.eigenvalues.slice(s![..k]).to_owned();
        let s = subspace_eig.eigenvectors.slice(s![..m, ..k]).to_owned();
        
        // Compute Ritz vectors: y_i = V * s_i
        let y = v_basis.dot(&s);
        
        // Compute residuals: r_i = A*y_i - theta_i*y_i
        let mut residuals = Array2::zeros((n, k));
        let mut residual_norms = Array1::zeros(k);
        
        for i in 0..k {
            let yi = y.column(i);
            operator.apply(&yi.to_owned(), &mut av);
            let ri = &av - &(&yi * theta[i]);
            residuals.column_mut(i).assign(&ri);
            
            let r_norm = ri.dot(&ri).sqrt();
            residual_norms[i] = r_norm;
        }
        
        // Check convergence
        max_residual = residual_norms.fold(0.0, |max, &val| max.max(val));
        converged = max_residual < config.tolerance;
        
        if converged || iteration == config.max_iter - 1 {
            // Return the result if converged or reached max iterations
            return DavidsonResult {
                eigenvalues: theta,
                eigenvectors: y,
                iterations: iteration + 1,
                converged,
                max_residual,
            };
        }
        
        // Expand the subspace with preconditioned residuals
        let mut new_vectors = Vec::new();
        
        for i in 0..k {
            if residual_norms[i] > config.tolerance {
                let ri = residuals.column(i);
                let mut qi = ri.to_owned();
                
                // Apply preconditioning if enabled
                if config.use_preconditioning {
                    for j in 0..n {
                        let denom = diag[j] - theta[i];
                        if denom.abs() > 1e-10 {
                            qi[j] /= denom;
                        } else {
                            qi[j] /= 1e-10; // Avoid division by zero
                        }
                    }
                }
                
                // Orthogonalize against the current basis (Modified Gram-Schmidt)
                for j in 0..m {
                    let vj = v_basis.column(j);
                    let projection = vj.dot(&qi);
                    qi = &qi - &(&vj * projection);
                }
                
                // Normalize the new vector
                let norm = qi.dot(&qi).sqrt();
                if norm > 1e-10 {
                    qi /= norm;
                    new_vectors.push(qi);
                }
            }
        }
        
        // Add new vectors to the basis
        if !new_vectors.is_empty() {
            let new_m = m + new_vectors.len();
            let mut new_basis = Array2::zeros((n, new_m));
            
            // Copy existing basis
            for j in 0..m {
                new_basis.column_mut(j).assign(&v_basis.column(j));
            }
            
            // Add new vectors
            for (j, q) in new_vectors.iter().enumerate() {
                new_basis.column_mut(m + j).assign(q);
            }
            
            v_basis = new_basis;
        }
        
        // Check if we need to restart (subspace size exceeds max_subspace_size)
        if v_basis.shape()[1] > config.max_subspace_size {
            // Restart with the current Ritz vectors plus some additional vectors
            let mut new_basis = Array2::zeros((n, config.min_subspace_size));
            
            // Use the current Ritz vectors as the new basis
            for j in 0..k {
                new_basis.column_mut(j).assign(&y.column(j));
            }
            
            // Add some of the most recent correction vectors
            let remaining = config.min_subspace_size - k;
            if remaining > 0 {
                let start_idx = v_basis.shape()[1] - remaining;
                for j in 0..remaining {
                    new_basis.column_mut(k + j).assign(&v_basis.column(start_idx + j));
                }
            }
            
            // Reorthogonalize the basis using Gram-Schmidt
            for j in 0..config.min_subspace_size {
                let mut vj = new_basis.column(j).to_owned();
                
                // Orthogonalize against previous vectors
                for i in 0..j {
                    let vi = new_basis.column(i);
                    let projection = vi.dot(&vj);
                    vj = &vj - &(&vi * projection);
                }
                
                // Normalize
                let norm = vj.dot(&vj).sqrt();
                if norm > 1e-10 {
                    vj /= norm;
                    new_basis.column_mut(j).assign(&vj);
                }
            }
            
            v_basis = new_basis;
            
            // Resize the subspace matrix
            subspace_matrix = Array2::zeros((config.min_subspace_size, config.min_subspace_size));
        }
        
        iteration += 1;
    }
    
    // If we reach here, we've hit the maximum number of iterations without converging
    // Compute the final Ritz vectors and return
    let m = v_basis.shape()[1];
    let subspace_eig = subspace_matrix.slice(s![0..m, 0..m]).eigh(UPLO::Upper).unwrap();
    let theta = subspace_eig.eigenvalues.slice(s![..k]).to_owned();
    let s = subspace_eig.eigenvectors.slice(s![..m, ..k]).to_owned();
    let y = v_basis.dot(&s);
    
    DavidsonResult {
        eigenvalues: theta,
        eigenvectors: y,
        iterations: iteration,
        converged: false,
        max_residual,
    }
}

// Example implementation of LinearOperator for a simple test case
struct MatrixFreeOperator {
    dim: usize,
    diag: Array1<f64>,
    offdiag: f64,
}

impl MatrixFreeOperator {
    fn new(dim: usize, diag_value: f64, offdiag: f64) -> Self {
        let mut diag = Array1::zeros(dim);
        for i in 0..dim {
            diag[i] = diag_value + i as f64;
        }
        Self { dim, diag, offdiag }
    }
}

impl LinearOperator for MatrixFreeOperator {
    fn apply(&self, v: &Array1<f64>, result: &mut Array1<f64>) {
        // Apply diagonal part
        for i in 0..self.dim {
            result[i] = self.diag[i] * v[i];
        }
        
        // Apply off-diagonal part (tridiagonal in this example)
        for i in 1..self.dim {
            result[i] += self.offdiag * v[i-1];
            result[i-1] += self.offdiag * v[i];
        }
    }
    
    fn dim(&self) -> usize {
        self.dim
    }
    
    fn diagonal(&self) -> Array1<f64> {
        self.diag.clone()
    }
}

// Example usage
fn main() {
    // Create a tridiagonal operator
    let dim = 1000;
    let operator = MatrixFreeOperator::new(dim, 2.0, -1.0);
    
    // Configure Davidson algorithm
    let config = DavidsonConfig {
        max_iter: 100,
        max_subspace_size: 30,
        min_subspace_size: 15,
        tolerance: 1e-8,
        n_eigenvalues: 5,
        use_preconditioning: true,
    };
    
    // Run Davidson algorithm
    let result = davidson(&operator, &config, None);
    
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("Max residual: {:.2e}", result.max_residual);
    println!("Eigenvalues: {:?}", result.eigenvalues);
}
