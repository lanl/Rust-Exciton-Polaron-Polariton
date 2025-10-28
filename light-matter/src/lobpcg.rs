/// Depreciated --- Use ndarray_linalg::lobpcg instead

use ndarray::{Array1, Array2, s};
use ndarray_linalg::{EighInto, Norm, UPLO, QRSquareInplace};
use std::fmt;
use rand::Rng;
use rayon::prelude::*;

use crate::davidson::DavidsonOperator;

/// Configuration options for the LOBPCG algorithm
pub struct LobpcgConfig {
    pub max_iter: usize,
    pub tolerance: f64,
    pub num_eigenvalues: usize,
    pub initial_guess: Option<Array2<f64>>,
}

/// Enum for LOBPCG solver status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LobpcgStatus {
    NotStarted,
    Running,
    Converged,
    MaxIterationsReached,
    Failed,
}

/// Results of the LOBPCG solver
pub struct LobpcgResult {
    pub values: Vec<f64>,
    pub vectors: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
    pub residuals: Vec<f64>,
    pub status: LobpcgStatus,
    pub error: Option<String>,
}

impl fmt::Debug for LobpcgResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LobpcgResult {{ converged: {}, iterations: {}, status: {:?}, values: [{}...], residuals: [{}...] }}",
            self.converged,
            self.iterations,
            self.status,
            self.values.iter().take(3).map(|v| format!("{:.6}", v)).collect::<Vec<_>>().join(", "),
            self.residuals.iter().take(3).map(|v| format!("{:.6e}", v)).collect::<Vec<_>>().join(", ")
        )
    }
}

/// Main LOBPCG solver function
pub fn lobpcg<T: DavidsonOperator<f64> + Sync>(operator: &T, config: &LobpcgConfig) -> LobpcgResult {
    let n = operator.dimension();
    let m = config.num_eigenvalues;
    
    if m > n {
        return LobpcgResult {
            values: Vec::new(),
            vectors: Vec::new(),
            converged: false,
            iterations: 0,
            residuals: Vec::new(),
            status: LobpcgStatus::Failed,
            error: Some("Number of requested eigenvalues cannot exceed matrix dimension".to_string()),
        };
    }
    
    // Initialize block of vectors (either from initial guess or random)
    let mut x = if let Some(ref initial_guess) = config.initial_guess {
        if initial_guess.nrows() != n || initial_guess.ncols() < m {
            return LobpcgResult {
                values: Vec::new(),
                vectors: Vec::new(),
                converged: false,
                iterations: 0,
                residuals: Vec::new(),
                status: LobpcgStatus::Failed,
                error: Some(format!("Initial guess has wrong dimensions: expected ({}, {}), got ({}, {})", 
                                  n, m, initial_guess.nrows(), initial_guess.ncols())),
            };
        }
        // Use the provided initial guess
        if initial_guess.ncols() > m {
            // Take only the requested number of columns
            initial_guess.slice(s![.., ..m]).to_owned()
        } else {
            initial_guess.clone()
        }
    } else {
        // Generate random initial vectors
        let mut x = Array2::<f64>::zeros((n, m));
        let mut rng = rand::rng();
        
        // Fill with random values
        for i in 0..m {
            for j in 0..n {
                x[[j, i]] = rng.random_range(-1.0..1.0);
            }
        }
        x
    };
    
    // Orthonormalize the initial block using QR decomposition
    orthonormalize_block(&mut x);
    
    // Get diagonal for preconditioning
    let diag_orig = Array1::from(operator.get_diagonal());
    
    // Apply shift-and-invert technique to improve convergence
    // Find a suitable shift value (typically larger than the largest expected eigenvalue)
    let shift = diag_orig.iter().fold(0.0, |max_val: f64, &val: &f64| max_val.max(val.abs())) * 1.1;
    
    // Create shifted diagonal for preconditioning
    let diag = &diag_orig + shift;
    
    // Initialize work arrays
    let mut p = Array2::<f64>::zeros((n, m));
    let mut r = Array2::<f64>::zeros((n, m));
    let mut w = Array2::<f64>::zeros((n, m));
    
    // Arrays for the small generalized eigenvalue problem
    let mut a_rayleigh = Array2::<f64>::zeros((3*m, 3*m));
    let mut b_rayleigh = Array2::<f64>::zeros((3*m, 3*m));
    
    // Create temporary arrays for matrix-vector products
    let mut ax = Array2::<f64>::zeros((n, m));
    let mut ap = Array2::<f64>::zeros((n, m));
    
    // Arrays for eigenvalues and residual norms
    let mut lambda = Array1::<f64>::zeros(m);
    let mut res_norms = Array1::<f64>::zeros(m);
    
    // Variables for three-term recurrence
    let mut x_prev: Option<Array2<f64>> = None;
    let mut ax_prev: Option<Array2<f64>> = None;
    
    // Apply operator to initial block: Ax
    apply_operator_to_block(operator, &x, &mut ax);
    
    // Calculate initial residuals: R = AX - XΛ
    calculate_residuals(&x, &ax, &mut r, &mut lambda);
    
    // Initial preconditioned residuals
    precondition(&r, &diag, &mut p);
    
    // Initialize status
    let mut status = LobpcgStatus::Running;
    let mut iterations = 0;
    
    // Track which eigenvalues are still active (not converged)
    let mut activemask = vec![true; m];
    let mut previous_block_size = m;
    
    // Track the best result seen so far
    let mut best_result: Option<(Array1<f64>, Array2<f64>, Vec<f64>)> = None;
    
    // Flag to control explicit Gram matrix calculation
    let mut _explicit_gram_flag = true;
    
    // Store previous P and AP for 3-term recurrence
    let mut _previous_p_ap: Option<(Array2<f64>, Array2<f64>)> = None;
    
    // Main iteration loop
    while iterations < config.max_iter {
        iterations += 1;
        
        // Apply operator to P: AP
        apply_operator_to_block(operator, &p, &mut ap);
        
        if iterations > 1 {
            // Form the Rayleigh quotient matrix for the subspace [X P X_prev]
            form_rayleigh_matrices(&x, &p, x_prev.as_ref(), &ax, &ap, ax_prev.as_ref(), &mut a_rayleigh, &mut b_rayleigh, 3*m);
        } else {
            // First iteration, use only [X P]
            form_rayleigh_matrices(&x, &p, None, &ax, &ap, None, &mut a_rayleigh, &mut b_rayleigh, 2*m);
        }
        
        // Solve the generalized eigenvalue problem using ndarray_linalg
        // First try with the current subspace, if that fails, try with a smaller subspace
        let eigenpairs = match solve_generalized_eigenproblem(&a_rayleigh, &b_rayleigh, m) {
            Ok(pairs) => pairs,
            Err(e) => {
                // If we're using 3-term recurrence, fall back to 2-term
                if iterations > 1 {
                    println!("Warning: 3-term recurrence failed, falling back to 2-term recurrence");
                    form_rayleigh_matrices(&x, &p, None, &ax, &ap, None, &mut a_rayleigh, &mut b_rayleigh, 2*m);
                    match solve_generalized_eigenproblem(&a_rayleigh, &b_rayleigh, m) {
                        Ok(pairs) => pairs,
                        Err(_) => {
                            // If that still fails, try with regularization
                            println!("Warning: 2-term recurrence failed, trying with regularization");
                            // Add small regularization to diagonal of B
                            for i in 0..2*m {
                                b_rayleigh[[i, i]] += 1e-10;
                            }
                            match solve_generalized_eigenproblem(&a_rayleigh, &b_rayleigh, m) {
                                Ok(pairs) => pairs,
                                Err(_) => {
                                    // If all attempts fail, return the best result so far
                                    return LobpcgResult {
                                        values: lambda.to_vec(),
                                        vectors: x.as_slice().unwrap().to_vec(),
                                        converged: false,
                                        iterations,
                                        residuals: res_norms.to_vec(),
                                        status: LobpcgStatus::Failed,
                                        error: Some(format!("Error solving generalized eigenvalue problem: {}", e)),
                                    };
                                }
                            }
                        }
                    }
                } else {
                    // First iteration, try with regularization
                    println!("Warning: Initial eigenvalue problem failed, trying with regularization");
                    // Add small regularization to diagonal of B
                    for i in 0..2*m {
                        b_rayleigh[[i, i]] += 1e-10;
                    }
                    match solve_generalized_eigenproblem(&a_rayleigh, &b_rayleigh, m) {
                        Ok(pairs) => pairs,
                        Err(_) => {
                            // If all attempts fail, return the best result so far
                            return LobpcgResult {
                                values: lambda.to_vec(),
                                vectors: x.as_slice().unwrap().to_vec(),
                                converged: false,
                                iterations,
                                residuals: res_norms.to_vec(),
                                status: LobpcgStatus::Failed,
                                error: Some(format!("Error solving generalized eigenvalue problem: {}", e)),
                            };
                        }
                    }
                }
            }
        };
        
        // Save current X and AX for the next iteration
        if iterations > 1 {
            // We already have x_prev and ax_prev from the previous iteration
            update_vectors(&mut x, &mut p, &eigenpairs, x_prev.as_ref(), 3*m);
        } else {
            // First iteration, no x_prev yet
            update_vectors(&mut x, &mut p, &eigenpairs, None, 2*m);
        }
        
        // Store current X and AX for the next iteration
        x_prev = Some(x.clone());
        ax_prev = Some(ax.clone());
        
        // Apply operator to updated X: AX
        apply_operator_to_block(operator, &x, &mut ax);
        
        // Calculate residuals: R = AX - XΛ
        calculate_residuals(&x, &ax, &mut r, &mut lambda);
        
        // Calculate residual norms and update best result
        let mut max_res_norm = 0.0;
        for i in 0..m {
            res_norms[i] = r.column(i).norm();
            if res_norms[i] > max_res_norm {
                max_res_norm = res_norms[i];
            }
        }
        
        // Compare best result and update if we improved
        let sum_rnorm: f64 = res_norms.iter().sum();
        if best_result
            .as_ref()
            .map(|x: &(_, _, Vec<f64>)| x.2.iter().sum::<f64>() > sum_rnorm)
            .unwrap_or(true)
        {
            best_result = Some((lambda.clone(), x.clone(), res_norms.to_vec()));
        }
        
        // Update active mask - which eigenvalues are still not converged
        activemask = res_norms
            .iter()
            .zip(activemask.iter())
            .map(|(x, a)| *x > config.tolerance && *a)
            .collect();
            
        // Resize active block if necessary
        let current_block_size = activemask.iter().filter(|x| **x).count();
        if current_block_size != previous_block_size {
            previous_block_size = current_block_size;
        }
        
        // If all eigenvalues have converged or we've reached max iterations, exit
        if current_block_size == 0 {
            status = LobpcgStatus::Converged;
            break;
        }
        
        // Precondition residuals
        precondition(&r, &diag, &mut w);
        
        // Orthogonalize against current subspace
        orthogonalize_against_x(&mut w, &x);
        
        // Orthonormalize the block
        orthonormalize_block(&mut w);
        
        // Cycle vectors for next iteration
        std::mem::swap(&mut p, &mut w);
    }
    
    // Final status check
    if status != LobpcgStatus::Converged {
        status = LobpcgStatus::MaxIterationsReached;
    }
    
    // Use the best result we found during iterations
    let (best_lambda, best_x, best_residuals) = best_result.unwrap_or((lambda.clone(), x.clone(), res_norms.to_vec()));
    
    // Prepare result - subtract the shift to get the original eigenvalues
    let mut values = vec![0.0; m];
    for i in 0..m {
        values[i] = best_lambda[i] - shift;
    }
    
    // Flatten eigenvectors
    let mut vectors = vec![0.0; n * m];
    for i in 0..m {
        for j in 0..n {
            vectors[i * n + j] = best_x[[j, i]];
        }
    }
    
    // Convert residual norms to Vec
    let residuals = best_residuals;
    
    LobpcgResult {
        values,
        vectors,
        converged: status == LobpcgStatus::Converged,
        iterations,
        residuals,
        status,
        error: None,
    }
}

/// Orthonormalize a block of vectors using QR decomposition
fn orthonormalize_block(block: &mut Array2<f64>) {
    // Handle the case where the block might be rank-deficient
    let n = block.nrows();
    let m = block.ncols();
    
    // Try QR decomposition
    match block.qr_square_inplace() {
        Ok(_) => {
            // QR decomposition successful, block now contains Q
        },
        Err(_) => {
            // QR failed, likely due to rank deficiency
            // Fall back to a more robust approach
            let mut rng = rand::rng();
            
            // First, check for and fix any near-zero columns
            let mut fixed_any = false;
            
            // Check norms of columns
            let mut norms = vec![0.0; m];
            for i in 0..m {
                norms[i] = block.column(i).norm();
            }
            
            // Fix any near-zero columns
            for i in 0..m {
                if norms[i] < 1e-12 {
                    fixed_any = true;
                    
                    // Create a random vector
                    let mut col_i = Array1::<f64>::zeros(n);
                    for k in 0..n {
                        col_i[k] = rng.random_range(-1.0..1.0);
                    }
                    
                    // Orthogonalize against previous columns
                    for j in 0..i {
                        let col_j = block.column(j).to_owned();
                        let dot = col_i.dot(&col_j);
                        col_i = &col_i - &(&col_j * dot);
                    }
                    
                    // Normalize
                    let norm = col_i.norm();
                    if norm > 1e-12 {
                        col_i = &col_i / norm;
                    } else {
                        // If still too small, set to zeros
                        col_i.fill(0.0);
                    }
                    
                    // Copy back to the block
                    for k in 0..n {
                        block[[k, i]] = col_i[k];
                    }
                }
            }
            
            // If we fixed any columns, try QR again
            if fixed_any {
                match block.qr_square_inplace() {
                    Ok(_) => {
                        // QR decomposition successful, block now contains Q
                        return;
                    },
                    Err(_) => {
                        // Continue to Gram-Schmidt
                    }
                }
            }
            
            // If still failing, use modified Gram-Schmidt as last resort
            // This is more numerically stable than classical Gram-Schmidt
            for i in 0..m {
                // Normalize the current column
                let mut col_i = block.column_mut(i);
                let norm_i = col_i.norm();
                
                if norm_i > 1e-12 {
                    col_i.mapv_inplace(|x| x / norm_i);
                } else {
                    col_i.fill(0.0);
                    continue;
                }
                
                // Orthogonalize all subsequent columns against this one
                for j in (i+1)..m {
                    // Store the dot product and column i values
                    let dot = block.column(i).dot(&block.column(j));
                    let col_i_values: Vec<f64> = block.column(i).iter().cloned().collect();
                    
                    // Now we can modify column j without borrowing block twice
                    let mut col_j = block.column_mut(j);
                    
                    // col_j -= dot * col_i
                    for k in 0..n {
                        col_j[k] -= dot * col_i_values[k];
                    }
                }
            }
        }
    }
}

/// Orthogonalize a block against another block
fn orthogonalize_against_x(block: &mut Array2<f64>, x: &Array2<f64>) {
    let n = block.nrows();
    let m_block = block.ncols();
    let m_x = x.ncols();
    
    // For each column in the block
    for i in 0..m_block {
        let mut col_i = block.column(i).to_owned();
        
        // Orthogonalize against each column in x
        for j in 0..m_x {
            let x_j = x.column(j).to_owned();
            let dot = col_i.dot(&x_j);
            col_i = &col_i - &(&x_j * dot);
        }
        
        // Copy back to the block
        for k in 0..n {
            block[[k, i]] = col_i[k];
        }
    }
}

/// Apply operator to a block of vectors
fn apply_operator_to_block<T: DavidsonOperator<f64> + Sync>(
    operator: &T,
    input_block: &Array2<f64>,
    output_block: &mut Array2<f64>
) {
    let n = input_block.nrows();
    let m = input_block.ncols();
    
    // If we can't use parallelism, fall back to sequential processing
    if m <= 1 {
        // Create temporary vectors for operator application
        let mut x_temp = vec![0.0; n];
        let mut y_temp = vec![0.0; n];
        
        for i in 0..m {
            // Copy column to temporary vector
            for j in 0..n {
                x_temp[j] = input_block[[j, i]];
            }
            
            // Apply operator
            operator.apply(&x_temp, &mut y_temp);
            
            // Copy result back to output block
            for j in 0..n {
                output_block[[j, i]] = y_temp[j];
            }
        }
    } else {
        // Process columns in parallel using thread-safe approach
        let results: Vec<_> = (0..m).into_par_iter().map(|i| {
            // Create temporary vectors for operator application
            let mut x_temp = vec![0.0; n];
            let mut y_temp = vec![0.0; n];
            
            // Copy column to temporary vector
            for j in 0..n {
                x_temp[j] = input_block[[j, i]];
            }
            
            // Apply operator
            operator.apply(&x_temp, &mut y_temp);
            
            // Return the result for this column
            (i, y_temp)
        }).collect();
        
        // Now copy all results back to the output block
        for (i, y_temp) in results {
            for j in 0..n {
                output_block[[j, i]] = y_temp[j];
            }
        }
    }
}

/// Calculate residuals R = AX - XΛ and eigenvalues
fn calculate_residuals(
    x: &Array2<f64>,
    ax: &Array2<f64>,
    r: &mut Array2<f64>,
    lambda: &mut Array1<f64>
) {
    let m = x.ncols();
    let n = x.nrows();
    
    // Calculate Rayleigh quotients (eigenvalues) in parallel
    let lambda_values: Vec<_> = (0..m).into_par_iter().map(|i| {
        let x_i = x.column(i);
        let ax_i = ax.column(i);
        ax_i.dot(&x_i)
    }).collect();
    
    // Update lambda array
    for i in 0..m {
        lambda[i] = lambda_values[i];
    }
    
    // Calculate residuals: R = AX - XΛ in parallel
    let residuals: Vec<_> = (0..m).into_par_iter().map(|i| {
        let lambda_i = lambda[i];
        let x_i = x.column(i);
        let ax_i = ax.column(i);
        
        // r_i = ax_i - lambda_i * x_i
        let mut r_i = vec![0.0; n];
        for j in 0..n {
            r_i[j] = ax_i[j] - lambda_i * x_i[j];
        }
        (i, r_i)
    }).collect();
    
    // Update r matrix
    for (i, r_i) in residuals {
        for j in 0..n {
            r[[j, i]] = r_i[j];
        }
    }
}

/// Precondition residuals using diagonal preconditioning
fn precondition(
    r: &Array2<f64>,
    diag: &Array1<f64>,
    p: &mut Array2<f64>
) {
    let n = r.nrows();
    let m = r.ncols();
    
    // Create safe diagonal values once (outside the loop)
    let safe_diag = diag.mapv(|x| x.abs().max(1e-10));
    
    // Process columns in parallel using thread-safe approach
    let preconditioned: Vec<_> = (0..m).into_par_iter().map(|i| {
        let r_col = r.column(i);
        
        // Apply diagonal preconditioner
        let mut p_col = vec![0.0; n];
        for j in 0..n {
            p_col[j] = r_col[j] / safe_diag[j];
        }
        (i, p_col)
    }).collect();
    
    // Update p matrix
    for (i, p_col) in preconditioned {
        for j in 0..n {
            p[[j, i]] = p_col[j];
        }
    }
}

/// Form Rayleigh quotient matrices for the subspace
fn form_rayleigh_matrices(
    x: &Array2<f64>,
    p: &Array2<f64>,
    x_prev: Option<&Array2<f64>>,  // Optional previous X
    ax: &Array2<f64>,
    ap: &Array2<f64>,
    ax_prev: Option<&Array2<f64>>, // Optional previous AX
    a_rayleigh: &mut Array2<f64>,
    b_rayleigh: &mut Array2<f64>,
    subspace_dim: usize
) {
    let m = x.ncols();
    
    // Clear the matrices
    a_rayleigh.fill(0.0);
    b_rayleigh.fill(0.0);
    
    // Fill the blocks of the Rayleigh matrices using matrix-matrix operations and slicing
    
    // X^T * AX block - using matrix multiplication and slicing
    let x_t_ax = x.t().dot(ax);
    a_rayleigh.slice_mut(s![0..m, 0..m]).assign(&x_t_ax);
    
    // X^T * AP block - using matrix multiplication and slicing
    let x_t_ap = x.t().dot(ap);
    a_rayleigh.slice_mut(s![0..m, m..2*m]).assign(&x_t_ap);
    a_rayleigh.slice_mut(s![m..2*m, 0..m]).assign(&x_t_ap.t()); // Symmetry
    
    // P^T * AP block - using matrix multiplication and slicing
    let p_t_ap = p.t().dot(ap);
    a_rayleigh.slice_mut(s![m..2*m, m..2*m]).assign(&p_t_ap);
    
    // Add third block if needed (for X_prev)
    if subspace_dim > 2*m && x_prev.is_some() && ax_prev.is_some() {
        let x_prev = x_prev.unwrap();
        let ax_prev = ax_prev.unwrap();
        
        // X^T * AX_prev
        let x_t_ax_prev = x.t().dot(ax_prev);
        a_rayleigh.slice_mut(s![0..m, 2*m..3*m]).assign(&x_t_ax_prev);
        a_rayleigh.slice_mut(s![2*m..3*m, 0..m]).assign(&x_t_ax_prev.t()); // Symmetry
        
        // P^T * AX_prev
        let p_t_ax_prev = p.t().dot(ax_prev);
        a_rayleigh.slice_mut(s![m..2*m, 2*m..3*m]).assign(&p_t_ax_prev);
        a_rayleigh.slice_mut(s![2*m..3*m, m..2*m]).assign(&p_t_ax_prev.t()); // Symmetry
        
        // X_prev^T * AX_prev
        let x_prev_t_ax_prev = x_prev.t().dot(ax_prev);
        a_rayleigh.slice_mut(s![2*m..3*m, 2*m..3*m]).assign(&x_prev_t_ax_prev);
    }
    
    // Build the mass matrix B (overlap matrix)
    
    // X^T * X block (should be identity)
    // Since X is orthonormal, X^T * X = I
    for i in 0..m {
        b_rayleigh[[i, i]] = 1.0;
    }
    
    // X^T * P block - using matrix multiplication and slicing
    let x_t_p = x.t().dot(p);
    b_rayleigh.slice_mut(s![0..m, m..2*m]).assign(&x_t_p);
    b_rayleigh.slice_mut(s![m..2*m, 0..m]).assign(&x_t_p.t()); // Symmetry
    
    // P^T * P block - using matrix multiplication and slicing
    let p_t_p = p.t().dot(p);
    b_rayleigh.slice_mut(s![m..2*m, m..2*m]).assign(&p_t_p);
    
    // Add third block if needed (for X_prev)
    if subspace_dim > 2*m && x_prev.is_some() {
        let x_prev = x_prev.unwrap();
        
        // X^T * X_prev
        let x_t_x_prev = x.t().dot(x_prev);
        b_rayleigh.slice_mut(s![0..m, 2*m..3*m]).assign(&x_t_x_prev);
        b_rayleigh.slice_mut(s![2*m..3*m, 0..m]).assign(&x_t_x_prev.t()); // Symmetry
        
        // P^T * X_prev
        let p_t_x_prev = p.t().dot(x_prev);
        b_rayleigh.slice_mut(s![m..2*m, 2*m..3*m]).assign(&p_t_x_prev);
        b_rayleigh.slice_mut(s![2*m..3*m, m..2*m]).assign(&p_t_x_prev.t()); // Symmetry
        
        // X_prev^T * X_prev (should be identity)
        for i in 0..m {
            b_rayleigh[[2*m + i, 2*m + i]] = 1.0;
        }
    }
}

/// Struct to hold eigenvalue and eigenvector from generalized eigenvalue problem
struct GeneralizedEigenpair {
    eigenvalue: f64,
    eigenvector: Array1<f64>,
}

/// Solve the small generalized eigenvalue problem using ndarray_linalg
fn solve_generalized_eigenproblem(a: &Array2<f64>, b: &Array2<f64>, num_eigs: usize) -> Result<Vec<GeneralizedEigenpair>, String> {
    // Use ndarray_linalg's generalized eigenvalue solver
    let n = a.nrows();
    let subspace_size = std::cmp::min(n, num_eigs);
    
    // Make sure matrices are symmetric
    let a_sym = make_symmetric(a);
    let mut b_sym = make_symmetric(b);
    
    // Add a small regularization to the diagonal of B to improve conditioning
    for i in 0..b_sym.nrows() {
        b_sym[[i, i]] += 1e-12 * (1.0 + b_sym[[i, i]].abs());
    }
    
    // Solve generalized eigenvalue problem Ax = λBx
    let (eigenvalues, (eigenvectors,_)) = match (a_sym, b_sym).eigh_into(UPLO::Upper) {
        Ok(result) => result,
        Err(e) => return Err(format!("Failed to solve generalized eigenvalue problem: {}", e)),
    };
    
    // Create result vector
    let mut result = Vec::with_capacity(subspace_size);
    for i in 0..subspace_size {
        let eigenvector = eigenvectors.column(i).to_owned();
        result.push(GeneralizedEigenpair {
            eigenvalue: eigenvalues[i],
            eigenvector,
        });
    }
    
    // Sort by eigenvalues (should already be sorted, but just to be sure)
    result.sort_by(|a, b| a.eigenvalue.partial_cmp(&b.eigenvalue).unwrap());
    
    Ok(result)
}

/// Make a matrix symmetric by averaging with its transpose
fn make_symmetric(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut result = Array2::<f64>::zeros((n, n));
    
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = 0.5 * (a[[i, j]] + a[[j, i]]);
        }
    }
    
    result
}

/// Update vectors based on the solution of the Rayleigh-Ritz procedure
fn update_vectors(
    x: &mut Array2<f64>,
    p: &mut Array2<f64>,
    eigenpairs: &[GeneralizedEigenpair],
    x_prev: Option<&Array2<f64>>,
    subspace_dim: usize
) {
    let n = x.nrows();
    let m = x.ncols();
    
    // Save old X and P
    let x_old = x.clone();
    let p_old = p.clone();
    
    // Update X with the new Ritz vectors
    for i in 0..m {
        let coefs = &eigenpairs[i].eigenvector;
        
        // Initialize new column
        let mut new_x = Array1::<f64>::zeros(n);
        
        // Add contribution from X
        for k in 0..m {
            let x_k = x_old.column(k).to_owned();
            new_x = &new_x + &(&x_k * coefs[k]);
        }
        
        // Add contribution from P
        for k in 0..m {
            if k + m < coefs.len() {
                let p_k = p_old.column(k).to_owned();
                new_x = &new_x + &(&p_k * coefs[k + m]);
            }
        }
        
        // Add contribution from X_prev if using 3-term recurrence
        if subspace_dim > 2*m && x_prev.is_some() {
            let x_prev = x_prev.unwrap();
            for k in 0..m {
                if k + 2*m < coefs.len() {
                    let x_prev_k = x_prev.column(k).to_owned();
                    new_x = &new_x + &(&x_prev_k * coefs[k + 2*m]);
                }
            }
        }
        
        // Copy to x
        for j in 0..n {
            x[[j, i]] = new_x[j];
        }
    }
}

/// Use LOBPCG to diagonalize a matrix using the DavidsonOperator trait
pub fn diagonalize_lobpcg<T: DavidsonOperator<f64> + Sync>(
    operator: &T, 
    n_eig: usize,
    initial_guess: Option<Array2<f64>>
) -> (Array1<f64>, Array2<f64>) {
    // Configure LOBPCG solver
    let config = LobpcgConfig {
        max_iter: 1000,
        tolerance: 1e-8,
        num_eigenvalues: n_eig,
        initial_guess,
    };
    
    // Solve eigenvalue problem using LOBPCG algorithm
    let result = lobpcg(operator, &config);
    
    // Check for errors
    if let Some(error) = &result.error {
        eprintln!("LOBPCG warning: {}", error);
    }
    
    if !result.converged {
        println!("Warning: LOBPCG algorithm did not converge after {} iterations", result.iterations);
        println!("Final residuals: {:?}", result.residuals);
    }
    
    // // Print convergence information
    // println!("Converged in {} iterations", result.iterations);

    // Convert to ndarray format
    let eigenvalues = Array1::from_vec(result.values);
    
    // Convert eigenvectors to ndarray format
    let n = operator.dimension();
    let mut eigenvectors = Array2::zeros((n, n_eig));
    
    for i in 0..n_eig {
        for j in 0..n {
            eigenvectors[[j, i]] = result.vectors[i * n + j];
        }
    }

    println!("Eigvals: {:?}", eigenvalues);
    
    (eigenvalues, eigenvectors)
}

/// Backward compatibility wrapper for the old function signature
pub fn diagonalize_lobpcg_simple<T: DavidsonOperator<f64> + Sync>(
    operator: &T, 
    n_eig: usize
) -> (Array1<f64>, Array2<f64>) {
    diagonalize_lobpcg(operator, n_eig, None)
}
