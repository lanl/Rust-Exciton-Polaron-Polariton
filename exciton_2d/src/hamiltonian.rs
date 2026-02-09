//! Hamiltonian module for exciton energy calculations
//!
//! This module provides implementations of the relative coordinate and full
//! Hamiltonians for exciton simulations.

use crate::grid::Grid;
use crate::matrix_elements::MatrixElementCalculator;

use log::{info, warn};
use ndarray::{Array1, Array2};
use ndarray_linalg::Eigh;
use num_complex::Complex64;
use std::collections::HashMap;
// use std::path::Path;
use std::time::Instant;
use ordered_float::OrderedFloat;

/// Hamiltonian for the relative coordinate motion of an exciton
///
/// This struct handles the electron-hole interaction in the relative coordinate frame.
/// It provides methods for solving the eigenvalue problem of the relative coordinate
/// Hamiltonian, which describes the internal motion of the electron-hole pair.
#[derive(Clone)]
pub struct RelativeHamiltonian {
    pub grid: Grid,
    pub mu: f64,
    pub save_dir: String,
    pub recalculate: bool,
    pub batch_size: usize,
    verbose: bool,
}

impl RelativeHamiltonian {
    /// Creates a new relative coordinate Hamiltonian
    ///
    /// # Arguments
    /// * `grid` - Spatial grid for discretization
    /// * `mu` - Reduced mass of the electron-hole pair
    /// * `save_dir` - Directory to save/load results
    /// * `recalculate` - Whether to force recalculation
    /// * `batch_size` - Batch size for memory-intensive operations
    /// * `verbose` - Whether to print detailed information
    pub fn new(grid: Grid, mu: f64, save_dir: String, recalculate: bool, batch_size: usize, verbose: bool) -> Self {
        Self {
            grid,
            mu,
            save_dir,
            recalculate,
            batch_size,
            verbose
        }
    }
    
    /// Solves the relative coordinate Hamiltonian
    ///
    /// This method computes the eigenvalues and eigenvectors of the relative coordinate
    /// Hamiltonian, either by loading from disk or calculating them directly using
    /// analytical solutions for the 2D hydrogen atom.
    ///
    /// # Arguments
    /// * `k_eigvals` - Number of eigenvalues to compute
    ///
    /// # Returns
    /// * Tuple containing eigenvalues, eigenvectors, and parity values
    pub fn solve(&self, k_eigvals: usize) -> (Array1<f64>, Array2<f64>, Array1<i32>) {
        // Check if we've already solved this
        let rel_filename = format!(
            "{}/rel_eigvals_grid{}_box{}_nrel{}_par.json",
            self.save_dir,
            self.grid.grid_size,
            self.grid.box_size,
            k_eigvals
        );
        
        // if Path::new(&rel_filename).exists() && !self.recalculate {
        //     if self.verbose {info!("Loading relative coordinate eigenstates from {}", rel_filename)};
        //     // In a real implementation, we'd deserialize from JSON
        //     // For now, we'll create ⋆⋆⋆⋆⋆ data
        //     let eigvals = Array1::zeros(k_eigvals);
        //     let eigvecs = Array2::zeros((self.grid.grid_size * self.grid.grid_size, k_eigvals));
        //     let parities = Array1::from_vec(vec![1; k_eigvals]);
        //     return (eigvals, eigvecs, parities);
        // }
        
        let start_time = Instant::now();
        if self.verbose {
            info!("Solving relative coordinate Hamiltonian...");
            info!("Grid size: {}x{}", self.grid.grid_size, self.grid.grid_size);
            info!("Box size: {}x{}", self.grid.box_size, self.grid.box_size);
            info!("Grid spacing: {:.6}", self.grid.dx);
        }
        
        // Use analytical solutions for the 2D hydrogen atom
        let (eigvals, eigvecs, parities) = self.solve_analytical(k_eigvals);
        
        if self.verbose { info!("Relative coordinate Hamiltonian solved in {:.2} seconds.",
              start_time.elapsed().as_secs_f64())};
        
        // Save the results
        if self.verbose {info!("Saving relative coordinate eigenstates to {}", rel_filename)};
        // In a real implementation, we'd serialize to JSON
        
        (eigvals, eigvecs, parities)
    }
    
    /// Solves the 2D hydrogen atom analytically
    ///
    /// This method uses analytical solutions for the 2D hydrogen atom to generate
    /// eigenstates on the numerical grid. It computes the eigenvalues, eigenvectors,
    /// and parity values for the specified number of states.
    ///
    /// The 2D hydrogen atom has energy levels E_n = -mu/(2*(n+1/2)^2) where n is the
    /// principal quantum number and mu is the reduced mass. The eigenfunctions are
    /// expressed in terms of associated Laguerre polynomials and spherical harmonics.
    ///
    /// # Arguments
    /// * `k_eigvals` - Number of eigenvalues to compute
    ///
    /// # Returns
    /// * Tuple containing eigenvalues, eigenvectors, and parity values
    fn solve_analytical(&self, k_eigvals: usize) -> (Array1<f64>, Array2<f64>, Array1<i32>) {
        if self.verbose {
            info!("Using analytical solutions for the 2D hydrogen atom...");
            info!("Note: Generating states without parity filtering (sine coupling mixes parities)");
        }
        
        // For 2D hydrogen atom, the energy levels are:
        // E_n = -mu/(2*(n+1/2)^2)
        // where n = 0, 1, 2, ... is the principal quantum number
        
        // Generate states (n, m) level by level
        // For each n, we have m = -n, -n+1, ..., n-1, n
        let mut states: Vec<(i32, i32)> = Vec::new();
        
        for n in 0.. {
            // For each n, we have m = -n, -n+1, ..., n-1, n
            for m in -n..=n {
                    states.push((n, m));
                if states.len() >= k_eigvals {
                    break;
                }
            }
            if states.len() >= k_eigvals {
                break;
            }
        }
        
        // Truncate to exactly k_eigvals states
        states.truncate(k_eigvals);
        
        // Calculate parity for each state (based on unperturbed wavefunction)
        // Note: This is the parity of the UNPERTURBED state, not the full eigenstate
        // With sine coupling, the full eigenstates won't have definite parity
        let parities = Array1::from_iter(
            states.iter().map(|(_, m)| if m.abs() % 2 == 0 { 1 } else { -1 })
        );
        
        // Calculate eigenvalues for each state
        let mut eigvals = Array1::zeros(k_eigvals);
        for (i, &(n, _)) in states.iter().enumerate() {
            eigvals[i] = -1.0 * self.mu / (2.0 * (n as f64 + 0.5).powi(2));
        }
        
        if self.verbose {info!("Analytical eigenvalues: {:?}", &eigvals.slice(ndarray::s![..5]))};
        
        // Now we need to represent the eigenfunctions on our grid
        // Process in batches to reduce memory usage
        let batch_size = self.batch_size;
        let mut eigvecs = Array2::zeros((self.grid.grid_size.pow(2), k_eigvals));
        
        // Get polar coordinates
        let r = self.grid.get_r();
        let theta = self.grid.get_theta();
        
            // Generate the eigenfunctions for each state in batches
            for batch_start in (0..k_eigvals).step_by(batch_size) {
                let batch_end = std::cmp::min(batch_start + batch_size, k_eigvals);
                let batch_states: &[(i32, i32)] = &states[batch_start..batch_end];
                if self.verbose {
                    info!("Processing batch {}/{}: states {} to {}",
                        batch_start / batch_size + 1,
                        (k_eigvals + batch_size - 1) / batch_size,
                        batch_start,
                        batch_end - 1);
                }
                
                // Process this batch of states
                for (i, &(n, m)) in batch_states.iter().enumerate() {
                    let idx = batch_start + i;
                    
                    // Characteristic length for this state
                    let n_r = n as f64 + 0.5;
                    let beta_n =  2.0* self.mu / n_r;
                    
                    // Scaled radius
                    let rho = &r * beta_n;
                    
                    // Compute the associated Laguerre polynomial
                    // For large grids, compute in blocks
                    let mut l_nm = Array2::zeros((self.grid.grid_size, self.grid.grid_size));
                    
                    if self.grid.grid_size > 128 {
                        let block_size = std::cmp::min(128, self.grid.grid_size);
                        
                        for i_block in (0..self.grid.grid_size).step_by(block_size) {
                            let i_end = std::cmp::min(i_block + block_size, self.grid.grid_size);
                            for j_block in (0..self.grid.grid_size).step_by(block_size) {
                                let j_end = std::cmp::min(j_block + block_size, self.grid.grid_size);
                                
                                // Extract block of rho
                                let rho_block = rho.slice(ndarray::s![i_block..i_end, j_block..j_end]);
                                
                                // Compute Laguerre polynomial for this block
                                for i in i_block..i_end {
                                    for j in j_block..j_end {
                                        l_nm[[i, j]] = self.eval_laguerre(n - m.abs(), 2 * m.abs(), rho_block[[i, j]]);
                                    }
                                }
                            }
                        }
                    } else {
                        // For smaller grids, compute directly
                        for i in 0..self.grid.grid_size {
                            for j in 0..self.grid.grid_size {
                                l_nm[[i, j]] = self.eval_laguerre(n - m.abs(), 2 * m.abs(), rho[[i, j]]);
                            }
                        }
                    }
                    
                    // Correct normalization constant for 2D hydrogen
                    let norm = beta_n * (self.factorial(n - m.abs()) as f64 / 
                              (self.factorial(n + m.abs()) as f64 * (2.0 * n_r))).sqrt();
                    
                    // Complete radial function
                    let mut radial = Array2::zeros((self.grid.grid_size, self.grid.grid_size));
                    for i in 0..self.grid.grid_size {
                        for j in 0..self.grid.grid_size {
                            radial[[i, j]] = norm * rho[[i, j]].powi(m.abs()) * (-rho[[i, j]] / 2.0).exp() * l_nm[[i, j]];
                        }
                    }
                    
                    // Angular function
                    let mut angular = Array2::zeros((self.grid.grid_size, self.grid.grid_size));
                    for i in 0..self.grid.grid_size {
                        for j in 0..self.grid.grid_size {
                            let phase = m as f64 * theta[[i, j]];
                            angular[[i, j]] = (phase.cos() + Complex64::i() * phase.sin()).re / (2.0 * std::f64::consts::PI).sqrt();
                        }
                    }
                    
                    // Complete wavefunction
                    let mut psi = Array2::zeros((self.grid.grid_size, self.grid.grid_size));
                    for i in 0..self.grid.grid_size {
                        for j in 0..self.grid.grid_size {
                            psi[[i, j]] = radial[[i, j]] * angular[[i, j]];
                        }
                    }
                    
                    // Store the eigenfunction
                    if m == 0 {
                        // For m=0, the wavefunction is real
                        for i in 0..self.grid.grid_size {
                            for j in 0..self.grid.grid_size {
                                eigvecs[[i * self.grid.grid_size + j, idx]] = psi[[i, j]];
                            }
                        }
                    } else {
                        // For m≠0, we can take linear combinations to get real functions
                        if m > 0 {
                            // For positive m, use cos(m*θ)
                            for i in 0..self.grid.grid_size {
                                for j in 0..self.grid.grid_size {
                                    let phase = m as f64 * theta[[i, j]];
                                    let real_part = radial[[i, j]] * phase.cos() / (2.0 * std::f64::consts::PI).sqrt();
                                    eigvecs[[i * self.grid.grid_size + j, idx]] = real_part / 2.0_f64.sqrt();
                                }
                            }
                        } else {
                            // For negative m, use sin(|m|*θ)
                            for i in 0..self.grid.grid_size {
                                for j in 0..self.grid.grid_size {
                                    let phase = m.abs() as f64 * theta[[i, j]];
                                    let imag_part = radial[[i, j]] * phase.sin() / (2.0 * std::f64::consts::PI).sqrt();
                                    eigvecs[[i * self.grid.grid_size + j, idx]] = imag_part / 2.0_f64.sqrt();
                                }
                            }
                        }
                    }
                }
            }
            
            // Normalize the eigenvectors on the grid
            for i in 0..k_eigvals {
                // For complex wavefunctions, we need |ψ|²
                let mut norm_squared = 0.0;
                for j in 0..eigvecs.nrows() {
                    norm_squared += eigvecs[[j, i]].powi(2);
                }
                norm_squared *= self.grid.dx.powi(2);
                
                let norm = norm_squared.sqrt();
                if norm > 1e-10 {  // Avoid division by zero
                    for j in 0..eigvecs.nrows() {
                        eigvecs[[j, i]] /= norm;
                    }
                }
            }
            
            if self.verbose {
                info!("Generated {} analytical eigenstates", k_eigvals);
                for i in 0..std::cmp::min(5, states.len()) {
                    let (n, m) = states[i];
                    info!("  State {}: n={}, m={}, E={:.6}", i, n, m, eigvals[i]);
                }
            }
            
            (eigvals, eigvecs, parities)
        }
        
    /// Evaluates the associated Laguerre polynomial L_n^(alpha)(x)
    ///
    /// # Arguments
    /// * `n` - Degree of the polynomial
    /// * `alpha` - Parameter of the associated Laguerre polynomial
    /// * `x` - Evaluation point
    ///
    /// # Returns
    /// * Value of the polynomial at x
    fn eval_laguerre(&self, n: i32, alpha: i32, x: f64) -> f64 {
        // Simple implementation of associated Laguerre polynomial
        // In a real implementation, we would use a more efficient algorithm
        let mut result = 0.0;
        for i in 0..=n {
            let binomial = self.binomial(n + alpha, n - i);
            result += binomial as f64 * (-x).powi(i) / self.factorial(i) as f64;
        }
        result
    }
    
    /// Computes the factorial of n
    ///
    /// # Arguments
    /// * `n` - Non-negative integer
    ///
    /// # Returns
    /// * n!
    fn factorial(&self, n: i32) -> u64 {
        if n <= 1 {
            return 1;
        }
        let mut result = 1;
        for i in 2..=n {
            result *= i as u64;
        }
        result
    }
    
    /// Computes the binomial coefficient (n choose k)
    ///
    /// # Arguments
    /// * `n` - Upper parameter
    /// * `k` - Lower parameter
    ///
    /// # Returns
    /// * Binomial coefficient n choose k
    fn binomial(&self, n: i32, k: i32) -> u64 {
        if k < 0 || k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }
        
        // Use symmetry to reduce computation
        let k = std::cmp::min(k, n - k);
        
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) as u64 / (i + 1) as u64;
        }
        result
    }
}

/// Full Hamiltonian for the exciton including center-of-mass motion
///
/// This struct combines the relative coordinate Hamiltonian with the center-of-mass
/// kinetic energy and the periodic potential. It provides methods for constructing
/// and diagonalizing the full Hamiltonian matrix, which describes both the internal
/// motion of the electron-hole pair and its motion through the periodic potential.
#[derive(Clone)]
pub struct FullHamiltonian {
    pub rel_ham: RelativeHamiltonian,
    pub matrix_elem_calc: MatrixElementCalculator,
    pub kappa_vectors: Vec<Vec<f64>>,
    pub vector_sum_to_index: HashMap<(usize, (OrderedFloat<f64>, OrderedFloat<f64>)), usize>,
    pub w_kappa: HashMap<(OrderedFloat<f64>, OrderedFloat<f64>), f64>,
    pub m: f64,
    pub k: Vec<f64>,
    pub verbose: bool,
}

impl FullHamiltonian {
    /// Creates a new full Hamiltonian
    ///
    /// # Arguments
    /// * `rel_ham` - Relative coordinate Hamiltonian
    /// * `matrix_elem_calc` - Matrix element calculator
    /// * `kappa_vectors` - Reciprocal lattice vectors
    /// * `vector_sum_to_index` - Lookup table for vector sums
    /// * `w_kappa` - Fourier coefficients of the potential
    /// * `m` - Total mass of the electron-hole pair
    /// * `k` - Wave vector
    /// * `verbose` - Whether to print detailed information
    pub fn new(
        rel_ham: RelativeHamiltonian,
        matrix_elem_calc: MatrixElementCalculator,
        kappa_vectors: Vec<Vec<f64>>,
        vector_sum_to_index: HashMap<(usize, (OrderedFloat<f64>, OrderedFloat<f64>)), usize>,
        w_kappa: HashMap<(OrderedFloat<f64>, OrderedFloat<f64>), f64>,
        m: f64,
        k: Vec<f64>,
        verbose: bool,
    ) -> Self {
        Self {
            rel_ham,
            matrix_elem_calc,
            kappa_vectors,
            vector_sum_to_index,
            w_kappa,
            m,
            k,
            verbose,
        }
    }
    
    /// Builds the Hamiltonian without center-of-mass kinetic energy
    ///
    /// This method constructs the Hamiltonian matrix that includes the relative coordinate
    /// energy and the periodic potential, but not the center-of-mass kinetic energy.
    /// With -i*sin(κ·r) matrix elements, the Hamiltonian is complex but Hermitian.
    ///
    /// # Arguments
    /// * `n_rel_states` - Number of relative coordinate states to include
    ///
    /// # Returns
    /// * Tuple containing the complex Hamiltonian matrix, relative eigenvalues, and relative eigenvectors
    pub fn build_hamiltonian_no_k(&self, n_rel_states: usize) -> (Array2<Complex64>, Array1<f64>, Array2<f64>) {
        // Solve for relative coordinate eigenstates
        let (rel_eigvals, rel_eigvecs, _parities) = self.rel_ham.solve(n_rel_states);
        
        // With sine matrix elements, we cannot filter by parity as even and odd states couple
        // Use all states
        let n_rel_states_used = rel_eigvals.len();
        
        if self.verbose {
            info!("Using all {} relative coordinate states (sine coupling mixes parities)", n_rel_states_used);
        }
        
        // Set the cached eigenvectors for matrix element calculations
        self.matrix_elem_calc.set_cached_eigvecs(rel_eigvecs.clone());
        
        // Get number of center-of-mass states
        let n_com_states = self.kappa_vectors.len();
        
        // Total dimension of the Hamiltonian
        let dim = n_rel_states_used * n_com_states;
        
        if self.verbose {
            info!("Building full Hamiltonian with dimension {}x{}", dim, dim);
            info!("Using {} relative states and {} COM states", n_rel_states_used, n_com_states);
        }
        
        // Convert w_kappa to use tuple keys and filter out zero values
        let mut w_kappa_tuples = HashMap::new();
        for (k, &v) in &self.w_kappa {
            if v.abs() > 1e-10 {  // Only keep non-zero values
                w_kappa_tuples.insert(*k, v);
            }
        }
        
        // Debug: Print number of non-zero potential terms
        if self.verbose {
            info!("Number of non-zero potential terms: {}", w_kappa_tuples.len());
            if w_kappa_tuples.is_empty() {
                warn!("WARNING: No non-zero potential terms found! Check w_kappa initialization.");
                // Print some of the kappa vectors to debug
                info!("Sample kappa vectors:");
                for i in 0..std::cmp::min(10, self.kappa_vectors.len()) {
                    info!("  {}: {:?}", i, self.kappa_vectors[i]);
                }
            }
        }
        
        // Calculate or load matrix elements
        let v_rel = self.matrix_elem_calc.calculate_matrix_elements(&w_kappa_tuples, n_rel_states_used);
        
        // Build the full Hamiltonian matrix
        if self.verbose {
            info!("Building the full Hamiltonian matrix...");
        }
        
        let start_time = Instant::now();
        
        // Create a dense complex matrix for the Hamiltonian
        let mut h_full = Array2::zeros((dim, dim));
        
        // Loop over all basis states
        for i_rel in 0..n_rel_states_used {
            for i_com in 0..n_com_states {
                let i = i_rel * n_com_states + i_com;
                
                // Add diagonal term (real)
                h_full[[i, i]] = Complex64::new(rel_eigvals[i_rel], 0.0);
                
                // Off-diagonal terms from the potential - only for non-zero w_kappa
                for (&kappa_pot_tuple, &v0) in &w_kappa_tuples {
                    // Find the index of the COM state with k' = k + kappa
                    let key = (i_com, kappa_pot_tuple);
                    if let Some(&j_com) = self.vector_sum_to_index.get(&key) {
                        // Loop over relative states
                        for j_rel in 0..n_rel_states_used {
                            let j = j_rel * n_com_states + j_com;
                            
                            // Get the potential matrix element if it exists
                            if let Some(&matrix_element) = v_rel.get(&(i_rel, j_rel, kappa_pot_tuple)) {
                                // Add to matrix if non-zero
                                // Matrix element is -i * sin(κ·r), so multiply by -i
                                if matrix_element.abs() > 1e-12 {
                                    h_full[[i, j]] += Complex64::new(0.0, -v0 * matrix_element);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if self.verbose {
            info!("Full Hamiltonian built in {:.2} seconds", start_time.elapsed().as_secs_f64());
            
            // Debug: Check if there are any off-diagonal elements
            let mut diag_sum = 0.0;
            let mut offdiag_sum = 0.0;
            
            for i in 0..dim {
                for j in 0..dim {
                    if i == j {
                        diag_sum += h_full[[i, j]].norm();
                    } else {
                        offdiag_sum += h_full[[i, j]].norm();
                    }
                }
            }
            
            info!("Sum of diagonal elements (magnitude): {}", diag_sum);
            info!("Sum of off-diagonal elements (magnitude): {}", offdiag_sum);
            if offdiag_sum < 1e-6 {
                warn!("WARNING: Almost no coupling between states!");
            }
        }
        
        (h_full, rel_eigvals, rel_eigvecs)
    }
    
    /// Builds the full Hamiltonian including center-of-mass kinetic energy
    ///
    /// This method constructs the complete Hamiltonian matrix that includes the relative
    /// coordinate energy, the periodic potential, and the center-of-mass kinetic energy.
    /// It adds the center-of-mass kinetic energy to the diagonal elements of the
    /// Hamiltonian matrix built by `build_hamiltonian_no_k`.
    ///
    /// # Arguments
    /// * `n_rel_states` - Number of relative coordinate states to include
    ///
    /// # Returns
    /// * Tuple containing the complex Hamiltonian matrix, relative eigenvalues, and relative eigenvectors
    pub fn build_full_hamiltonian(&self, n_rel_states: usize) -> (Array2<Complex64>, Array1<f64>, Array2<f64>) {
        // Build the Hamiltonian without center-of-mass kinetic energy
        let (mut h_full, rel_eigvals, rel_eigvecs) = self.build_hamiltonian_no_k(n_rel_states);
        
        // Number of center-of-mass states
        let n_com_states = self.kappa_vectors.len();
        let n_rel_states_used = rel_eigvals.len();
        
        // Add center-of-mass kinetic energy to diagonal elements
        for i_rel in 0..n_rel_states_used {
            for i_com in 0..n_com_states {
                let i = i_rel * n_com_states + i_com;
                
                // Diagonal term: relative energy + center-of-mass kinetic energy
                let k_vec_i = &self.kappa_vectors[i_com];
                let mut k_plus_k = vec![0.0; 2];
                for j in 0..2 {
                    k_plus_k[j] = self.k[j] + k_vec_i[j];
                }
                
                let e_com = (k_plus_k[0].powi(2) + k_plus_k[1].powi(2)) / (2.0 * self.m);
                h_full[[i, i]] += Complex64::new(e_com, 0.0);
            }
        }
        
        (h_full, rel_eigvals, rel_eigvecs)
    }
    
    /// Diagonalizes the full Hamiltonian
    ///
    /// This method diagonalizes the full Hamiltonian matrix to find the eigenvalues
    /// and eigenvectors of the exciton in the periodic potential.
    ///
    /// The Hamiltonian is complex (with -i*sin matrix elements) but Hermitian, so
    /// eigenvalues are real and eigenvectors are complex.
    ///
    /// # Arguments
    /// * `n_rel_states` - Number of relative coordinate states to include
    /// * `k_eigvals` - Number of eigenvalues to compute
    ///
    /// # Returns
    /// * Tuple containing eigenvalues (real), eigenvectors (complex), relative eigenvalues,
    ///   relative eigenvectors, and parity values (of unperturbed states)
    pub fn diagonalize_full_hamiltonian(
        &self,
        n_rel_states: usize,
        k_eigvals: usize,
    ) -> (Array1<f64>, Vec<Array1<Complex64>>, Array1<f64>, Array2<f64>) {
        if self.verbose {
            info!("Building complex Hermitian Hamiltonian with -i*sin matrix elements");
        }
        
        // Build the full Hamiltonian with all states
        let (h_full, rel_eigvals, rel_eigvecs) = self.build_full_hamiltonian(n_rel_states);
                
        let start_time = Instant::now();
        if self.verbose {
            info!("Diagonalizing full Hamiltonian with dimension {}x{}", h_full.nrows(), h_full.ncols());
            info!("Computing {} lowest eigenvalues...", k_eigvals);
        }
        
        // Compute eigenvalues and eigenvectors
        match h_full.eigh(ndarray_linalg::UPLO::Lower) {
            Ok((eigvals, eigvecs)) => {
                // Take only the first k_eigvals
                let n_eigvals = std::cmp::min(k_eigvals, eigvals.len());
                let eigvals = eigvals.slice(ndarray::s![..n_eigvals]).to_owned();
                
                // Convert eigenvectors to Vec<Array1<Complex64>> for convenience
                let mut eigvecs_list = Vec::new();
                for i in 0..n_eigvals {
                    eigvecs_list.push(eigvecs.slice(ndarray::s![.., i]).to_owned());
                }
                
                if self.verbose {
                    info!("Diagonalization completed in {:.2} seconds.", start_time.elapsed().as_secs_f64());
                    info!("First 5 eigenvalues: {:?}", eigvals.slice(ndarray::s![..std::cmp::min(5, n_eigvals)]));
                }
                
                (eigvals, eigvecs_list, rel_eigvals, rel_eigvecs)
            },
            Err(e) => {
                // Log detailed error information
                warn!("Eigenvalue computation failed for k = {:?}", self.k);
                warn!("Hamiltonian shape = {:?}", h_full.shape());
                warn!("Error details: {:?}", e);
                
                // Return empty results to avoid crashing
                let eigvals = Array1::zeros(k_eigvals);
                let mut eigvecs_list = Vec::new();
                for _ in 0..k_eigvals {
                    eigvecs_list.push(Array1::from_elem(h_full.nrows(), Complex64::new(0.0, 0.0)));
                }
                
                if self.verbose {
                    warn!("Using zeros as placeholder for failed diagonalization");
                }
                
                (eigvals, eigvecs_list, rel_eigvals, rel_eigvecs)
            }
        }
    }
}

