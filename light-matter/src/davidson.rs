// davidson.rs
use ndarray::{Array2};
use ndarray_linalg::eigh::Eigh;
use std::fmt::Debug;
use rand::Rng;

/// Trait for operators that can be used with the Davidson algorithm
pub trait DavidsonOperator<T: Clone + Debug> {
    /// Apply the operator to a vector
    fn apply(&self, x: &[T], y: &mut [T]);
    
    /// Get the diagonal of the operator (used for preconditioning)
    fn get_diagonal(&self) -> Vec<T>;
    
    /// Get the dimension of the operator
    fn dimension(&self) -> usize;
}

/// Configuration for the Davidson algorithm
pub struct DavidsonConfig {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Maximum subspace dimension before restart
    pub max_subspace_size: usize,
    /// Minimum subspace dimension after restart
    pub min_subspace_size: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Number of eigenvalues to compute
    pub num_eigenvalues: usize,
    /// Which eigenvalues to compute (smallest or largest)
    pub which: DavidsonWhich,
}

/// Which eigenvalues to compute
#[derive(Clone, Copy, Debug)]
pub enum DavidsonWhich {
    /// Smallest algebraic eigenvalues
    Smallest,
    /// Largest algebraic eigenvalues
    Largest,
}

impl Default for DavidsonConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            max_subspace_size: 20,
            min_subspace_size: 10,
            tolerance: 1e-6,
            num_eigenvalues: 1,
            which: DavidsonWhich::Smallest,
        }
    }
}

/// Result of the Davidson algorithm
pub struct DavidsonResult {
    /// Eigenvalues
    pub values: Vec<f64>,
    /// Eigenvectors
    pub vectors: Vec<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Residual norms for each eigenvalue
    pub residuals: Vec<f64>,
    /// Whether the algorithm converged
    pub converged: bool,
}

/// Davidson algorithm for finding a few eigenvalues and eigenvectors of a large sparse matrix
pub fn davidson<T>(
    op: &impl DavidsonOperator<T>,
    config: &DavidsonConfig,
) -> Result<DavidsonResult, String>
where
    T: Clone + Debug + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + 
       std::ops::Mul<Output = T> + std::ops::Div<Output = T> + 
       std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + 
       std::ops::DivAssign + PartialOrd + num_traits::Zero + num_traits::One + 
       num_traits::NumAssign + num_traits::Float + Copy + std::iter::Sum
{
    let n = op.dimension();
    let k = config.num_eigenvalues;
    
    if k >= n {
        return Err("Number of requested eigenvalues must be less than matrix dimension".to_string());
    }
    
    // Get diagonal for preconditioning
    let diag = op.get_diagonal();
    
    // Initialize subspace with random vectors
    let mut v = Array2::<T>::zeros((n, config.max_subspace_size));
    let mut rng = rand::rng();
    for j in 0..k {
        for i in 0..n {
            v[[i, j]] = T::from(rng.random::<f64>()).unwrap();
        }
        // Orthogonalize against previous vectors
        for l in 0..j {
            let dot_product = (0..n).map(|i| v[[i, j]] * v[[i, l]]).sum::<T>();
            let v_l = v.column(l).to_owned();
            for i in 0..n {
                v[[i, j]] -= dot_product * v_l[i];
            }
        }
        // Normalize
        let norm = (0..n).map(|i| v[[i, j]] * v[[i, j]]).sum::<T>().sqrt();
        for i in 0..n {
            v[[i, j]] /= norm;
        }
    }
    
    let mut subspace_size = k;
    let mut converged = false;
    let mut iterations = 0;
    
    // Allocate work arrays
    let mut av = Array2::<T>::zeros((n, config.max_subspace_size));
    let mut h = Array2::<T>::zeros((config.max_subspace_size, config.max_subspace_size));
    let mut s = Array2::<T>::zeros((config.max_subspace_size, k));
    let mut theta = Vec::<T>::with_capacity(k);
    let mut residuals = Vec::<T>::with_capacity(k);
    
    while iterations < config.max_iter && !converged {
        iterations += 1;
        
        // Apply operator to each basis vector
        for j in 0..subspace_size {
            let mut x_slice = vec![T::zero(); n];
            let mut y_slice = vec![T::zero(); n];
            
            for i in 0..n {
                x_slice[i] = v[[i, j]];
            }
            
            op.apply(&x_slice, &mut y_slice);
            
            for i in 0..n {
                av[[i, j]] = y_slice[i];
            }
        }
        
        // Form projected matrix H = V^T * A * V
        for i in 0..subspace_size {
            for j in 0..=i {
                h[[i, j]] = (0..n).map(|l| v[[l, i]] * av[[l, j]]).sum();
                if i != j {
                    h[[j, i]] = h[[i, j]];
                }
            }
        }
        
        // Solve the small eigenvalue problem
        let (eigenvalues, eigenvectors) = diagonalize_small_matrix(&h, subspace_size);
        
        // Sort eigenvalues and eigenvectors
        let mut indices: Vec<usize> = (0..subspace_size).collect();
        match config.which {
            DavidsonWhich::Smallest => {
                indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());
            }
            DavidsonWhich::Largest => {
                indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());
            }
        }
        
        // Extract the k eigenvalues and eigenvectors we want
        theta.clear();
        for &idx in indices.iter().take(k) {
            theta.push(eigenvalues[idx]);
        }
        
        // Compute Ritz vectors u = V * s
        let mut u = Array2::<T>::zeros((n, k));
        for l in 0..k {
            let idx = indices[l];
            for j in 0..subspace_size {
                s[[j, l]] = eigenvectors[[j, idx]];
            }
            
            for i in 0..n {
                for j in 0..subspace_size {
                    u[[i, l]] += v[[i, j]] * s[[j, l]];
                }
            }
        }
        
        // Compute residuals r = A*u - theta*u
        let mut r = Array2::<T>::zeros((n, k));
        residuals.clear();
        converged = true;
        
        for l in 0..k {
            let mut x_slice = vec![T::zero(); n];
            let mut y_slice = vec![T::zero(); n];
            
            for i in 0..n {
                x_slice[i] = u[[i, l]];
            }
            
            op.apply(&x_slice, &mut y_slice);
            
            for i in 0..n {
                r[[i, l]] = y_slice[i] - theta[l] * u[[i, l]];
            }
            
            let res_norm = (0..n).map(|i| r[[i, l]] * r[[i, l]]).sum::<T>().sqrt();
            residuals.push(res_norm);
            
            if res_norm > T::from(config.tolerance).unwrap() {
                converged = false;
            }
        }
        
        if converged {
            break;
        }
        
        // Check if we need to restart or expand the subspace
        if subspace_size >= config.max_subspace_size - k {
            // Restart with the current Ritz vectors
            // Copy u into the first k columns of v
            for j in 0..k {
                for i in 0..n {
                    v[[i, j]] = u[[i, j]];
                }
            }
            subspace_size = k;
        } else {
            // Expand the subspace with preconditioned residuals
            for l in 0..k {
                if residuals[l] > T::from(config.tolerance).unwrap() {
                    // Apply preconditioner: (diag - theta)^-1 * r
                    for i in 0..n {
                        let t = if (diag[i] - theta[l]).abs() > T::from(1e-10).unwrap() {
                            r[[i, l]] / (diag[i] - theta[l])
                        } else {
                            r[[i, l]] / T::from(1e-10).unwrap()
                        };
                        v[[i, subspace_size]] = t;
                    }
                    
                    // Orthogonalize against existing subspace
                    for j in 0..subspace_size {
                        let dot_product = (0..n).map(|i| v[[i, subspace_size]] * v[[i, j]]).sum::<T>();
                        let v_j = v.column(j).to_owned();
                        for i in 0..n {
                            v[[i, subspace_size]] -= dot_product * v_j[i];
                        }
                    }
                    
                    // Normalize
                    let norm = (0..n).map(|i| v[[i, subspace_size]] * v[[i, subspace_size]]).sum::<T>().sqrt();
                    if norm > T::from(1e-10).unwrap() {
                        for i in 0..n {
                            v[[i, subspace_size]] /= norm;
                        }
                        subspace_size += 1;
                    }
                    
                    if subspace_size >= config.max_subspace_size {
                        break;
                    }
                }
            }
        }
    }
    
    // Prepare result
    let mut result_values = Vec::with_capacity(k);
    let mut result_vectors = Vec::with_capacity(n * k);
    let mut result_residuals = Vec::with_capacity(k);

    for l in 0..k {
        result_values.push(theta[l].to_f64().unwrap());
        result_residuals.push(residuals[l].to_f64().unwrap());
        for i in 0..n {
            result_vectors.push(v[[i, l]].to_f64().unwrap());  // Use u for final vectors
        }
    }
    
    Ok(DavidsonResult {
        values: result_values,
        vectors: result_vectors,
        iterations,
        residuals: result_residuals,
        converged,
    })
}

// Helper function to diagonalize a small matrix using ndarray-linalg
fn diagonalize_small_matrix<T>(h: &Array2<T>, size: usize) -> (Vec<T>, Array2<T>)
where
    T: Clone + Debug + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + 
       std::ops::Mul<Output = T> + std::ops::Div<Output = T> + 
       std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + 
       std::ops::DivAssign + PartialOrd + num_traits::Zero + num_traits::One + 
       num_traits::NumAssign + num_traits::Float + Copy,
{
    // Convert to f64 for ndarray-linalg
    let mut h_f64 = Array2::<f64>::zeros((size, size));
    for i in 0..size {
        for j in 0..size {
            h_f64[[i, j]] = h[[i, j]].to_f64().unwrap();
        }
    }
    
    // Use ndarray-linalg's eigh for symmetric eigendecomposition
    let (eigenvalues, eigenvectors) = h_f64.eigh(ndarray_linalg::UPLO::Upper).unwrap();
    
    let mut eigenvalues_t = Vec::with_capacity(size);
    let mut eigenvectors_t = Array2::<T>::zeros((size, size));
    
    for i in 0..size {
        eigenvalues_t.push(T::from(eigenvalues[i]).unwrap());
        for j in 0..size {
            eigenvectors_t[[j, i]] = T::from(eigenvectors[[j, i]]).unwrap();
        }
    }
    
    (eigenvalues_t, eigenvectors_t)
}