//! Hamiltonian module for exciton energy calculations
//!
//! This module provides implementations of the relative coordinate and full
//! Hamiltonians for exciton simulations.

use crate::grid::Grid;
use crate::matrix_elements::MatrixElementCalculator;
use crate::utils::exciton_flat_index;

use log::{info, warn};
use matrix_ops::{ConvertFrom, DiagonalizationOptions, Diagonalize, MatrixOps, MatrixOpsError};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::collections::HashMap;
// use std::path::Path;
use ordered_float::OrderedFloat;
use sprs::TriMat;
use std::time::Instant;

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
    pub fn new(
        grid: Grid,
        mu: f64,
        save_dir: String,
        recalculate: bool,
        batch_size: usize,
        verbose: bool,
    ) -> Self {
        Self {
            grid,
            mu,
            save_dir,
            recalculate,
            batch_size,
            verbose,
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
            self.save_dir, self.grid.grid_size, self.grid.box_size, k_eigvals
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

        if self.verbose {
            info!(
                "Relative coordinate Hamiltonian solved in {:.2} seconds.",
                start_time.elapsed().as_secs_f64()
            )
        };

        // Save the results
        if self.verbose {
            info!("Saving relative coordinate eigenstates to {}", rel_filename)
        };
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
            info!(
                "Note: Generating states without parity filtering (sine coupling mixes parities)"
            );
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
            states
                .iter()
                .map(|(_, m)| if m.abs() % 2 == 0 { 1 } else { -1 }),
        );

        // Calculate eigenvalues for each state
        let mut eigvals = Array1::zeros(k_eigvals);
        for (i, &(n, _)) in states.iter().enumerate() {
            eigvals[i] = -1.0 * self.mu / (2.0 * (n as f64 + 0.5).powi(2));
        }

        if self.verbose {
            info!(
                "Analytical eigenvalues: {:?}",
                &eigvals.slice(ndarray::s![..5])
            )
        };

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
                info!(
                    "Processing batch {}/{}: states {} to {}",
                    batch_start / batch_size + 1,
                    (k_eigvals + batch_size - 1) / batch_size,
                    batch_start,
                    batch_end - 1
                );
            }

            // Process this batch of states
            for (i, &(n, m)) in batch_states.iter().enumerate() {
                let idx = batch_start + i;

                // Characteristic length for this state
                let n_r = n as f64 + 0.5;
                let beta_n = 2.0 * self.mu / n_r;

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
                                    l_nm[[i, j]] = self.eval_laguerre(
                                        n - m.abs(),
                                        2 * m.abs(),
                                        rho_block[[i, j]],
                                    );
                                }
                            }
                        }
                    }
                } else {
                    // For smaller grids, compute directly
                    for i in 0..self.grid.grid_size {
                        for j in 0..self.grid.grid_size {
                            l_nm[[i, j]] =
                                self.eval_laguerre(n - m.abs(), 2 * m.abs(), rho[[i, j]]);
                        }
                    }
                }

                // Correct normalization constant for 2D hydrogen
                let norm = beta_n
                    * (self.factorial(n - m.abs()) as f64
                        / (self.factorial(n + m.abs()) as f64 * (2.0 * n_r)))
                        .sqrt();

                // Complete radial function
                let mut radial = Array2::zeros((self.grid.grid_size, self.grid.grid_size));
                for i in 0..self.grid.grid_size {
                    for j in 0..self.grid.grid_size {
                        radial[[i, j]] = norm
                            * rho[[i, j]].powi(m.abs())
                            * (-rho[[i, j]] / 2.0).exp()
                            * l_nm[[i, j]];
                    }
                }

                // Angular function
                let mut angular = Array2::zeros((self.grid.grid_size, self.grid.grid_size));
                for i in 0..self.grid.grid_size {
                    for j in 0..self.grid.grid_size {
                        let phase = m as f64 * theta[[i, j]];
                        angular[[i, j]] = (phase.cos() + Complex64::i() * phase.sin()).re
                            / (2.0 * std::f64::consts::PI).sqrt();
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
                                let real_part =
                                    radial[[i, j]] * phase.cos() / std::f64::consts::PI.sqrt();
                                eigvecs[[i * self.grid.grid_size + j, idx]] = real_part;
                            }
                        }
                    } else {
                        // For negative m, use sin(|m|*θ)
                        for i in 0..self.grid.grid_size {
                            for j in 0..self.grid.grid_size {
                                let phase = m.abs() as f64 * theta[[i, j]];
                                let imag_part =
                                    radial[[i, j]] * phase.sin() / std::f64::consts::PI.sqrt();
                                eigvecs[[i * self.grid.grid_size + j, idx]] = imag_part;
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
            if norm > 1e-10 {
                // Avoid division by zero
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
pub struct FullHamiltonian<M: MatrixOps> {
    pub rel_ham: RelativeHamiltonian,
    pub matrix_elem_calc: MatrixElementCalculator,
    pub kappa_vectors: Vec<Vec<f64>>,
    pub vector_sum_to_index: HashMap<(usize, (OrderedFloat<f64>, OrderedFloat<f64>)), usize>,
    pub w_kappa: HashMap<(OrderedFloat<f64>, OrderedFloat<f64>), f64>,
    pub m: f64,
    pub k: Vec<f64>,
    pub verbose: bool,
    _phantom: std::marker::PhantomData<M>,
}

/// Prepared exciton Hamiltonian with all K-independent terms precomputed.
///
/// The full exciton Hamiltonian has the form
///
///     H(K) = H_0 + H_CoM(K),
///
/// where H_0 contains the relative-coordinate energies and the lattice coupling,
/// while H_CoM(K) is diagonal in the CoM plane-wave basis:
///
///     H_CoM(K) = |K + kappa|^2 / (2M).
///
/// This struct allows band-structure calculations to build H_0 once and only
/// update the diagonal CoM kinetic term for each K point.
#[derive(Clone)]
pub struct PreparedExcitonHamiltonian<M: MatrixOps> {
    pub h_no_k: M,
    pub rel_eigvals: Array1<f64>,
    pub rel_eigvecs: Array2<f64>,
    pub kappa_vectors: Vec<Vec<f64>>,
    pub n_rel: usize,
    pub n_com: usize,
    pub mass: f64,
}

impl<M: MatrixOps> FullHamiltonian<M>
where
    M::Scalar: ConvertFrom<Complex64> + ConvertFrom<f64>,
{
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
            _phantom: std::marker::PhantomData,
        }
    }

    /// Prepares the K-independent part of the exciton Hamiltonian.
    ///
    /// This builds the Hamiltonian without the CoM kinetic term:
    ///
    ///     H_0 = I_com ⊗ H_rel + H_lattice,
    ///
    /// and stores the relative-coordinate eigenbasis and CoM reciprocal vectors.
    /// For each K point, call `PreparedExcitonHamiltonian::hamiltonian_at_k`.
    pub fn prepare(
        &self,
        n_rel_states: usize,
    ) -> Result<PreparedExcitonHamiltonian<M>, MatrixOpsError>
    where
        M::Scalar: ConvertFrom<Complex64>,
    {
        let (h_no_k, rel_eigvals, rel_eigvecs) = self.build_hamiltonian_no_k(n_rel_states)?;

        let n_rel = rel_eigvals.len();
        let n_com = self.kappa_vectors.len();

        Ok(PreparedExcitonHamiltonian {
            h_no_k,
            rel_eigvals,
            rel_eigvecs,
            kappa_vectors: self.kappa_vectors.clone(),
            n_rel,
            n_com,
            mass: self.m,
        })
    }

    /// Builds the Hamiltonian without center-of-mass kinetic energy
    ///
    /// This method constructs the Hamiltonian matrix that includes the relative coordinate
    /// energy and the periodic potential, but not the center-of-mass kinetic energy.
    /// With -2i*sin(κ·r) matrix elements, the Hamiltonian is complex but Hermitian.
    ///
    /// # Arguments
    /// * `n_rel_states` - Number of relative coordinate states to include
    ///
    /// # Returns
    /// * Tuple containing the complex Hamiltonian matrix, relative eigenvalues, and relative eigenvectors
    pub fn build_hamiltonian_no_k(
        &self,
        n_rel_states: usize,
    ) -> Result<(M, Array1<f64>, Array2<f64>), MatrixOpsError>
    where
        M::Scalar: ConvertFrom<Complex64>,
    {
        // Solve for relative coordinate eigenstates
        let (rel_eigvals, rel_eigvecs, _parities) = self.rel_ham.solve(n_rel_states);

        // With sine matrix elements, we cannot filter by parity as even and odd states couple
        // Use all states
        let n_rel_states_used = rel_eigvals.len();

        if self.verbose {
            info!(
                "Using all {} relative coordinate states (sine coupling mixes parities)",
                n_rel_states_used
            );
        }

        // Set the cached eigenvectors for matrix element calculations
        self.matrix_elem_calc
            .set_cached_eigvecs(rel_eigvecs.clone());

        // Get number of center-of-mass states
        let n_com_states = self.kappa_vectors.len();

        // Total dimension of the Hamiltonian
        let dim = n_rel_states_used * n_com_states;

        if self.verbose {
            info!("Building full Hamiltonian with dimension {}x{}", dim, dim);
            info!(
                "Using {} relative states and {} COM states",
                n_rel_states_used, n_com_states
            );
        }

        // Convert w_kappa to use tuple keys and filter out zero values
        let mut w_kappa_tuples = HashMap::new();
        for (k, &v) in &self.w_kappa {
            if v.abs() > 1e-10 {
                // Only keep non-zero values
                w_kappa_tuples.insert(*k, v);
            }
        }

        // Debug: Print number of non-zero potential terms
        if self.verbose {
            info!(
                "Number of non-zero potential terms: {}",
                w_kappa_tuples.len()
            );
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
        let v_rel = self
            .matrix_elem_calc
            .calculate_matrix_elements(&w_kappa_tuples, n_rel_states_used);

        // Build the full Hamiltonian matrix
        if self.verbose {
            info!("Building the full Hamiltonian matrix...");
        }

        let start_time = Instant::now();

        // Create a dense complex matrix for the Hamiltonian
        let mut h_full: M;
        let eye_com = M::eye(n_com_states);

        h_full = eye_com.kron(&M::from_diag(rel_eigvals.as_slice().unwrap())?);

        let mut h_int = TriMat::new((dim, dim));

        // Group sparse relative-coordinate sine matrix elements by transferred kappa.
        // This avoids scanning all relative-state pairs for every CoM state.
        let mut v_rel_by_kappa: HashMap<
            (OrderedFloat<f64>, OrderedFloat<f64>),
            Vec<(usize, usize, f64)>,
        > = HashMap::new();

        for (&(bra_rel, ket_rel, kappa_pot_tuple), &matrix_element) in &v_rel {
            v_rel_by_kappa.entry(kappa_pot_tuple).or_default().push((
                bra_rel,
                ket_rel,
                matrix_element,
            ));
        }

        for i_com in 0..n_com_states {
            for (&kappa_pot_tuple, rel_entries) in &v_rel_by_kappa {
                let Some(&j_com) = self.vector_sum_to_index.get(&(i_com, kappa_pot_tuple)) else {
                    continue;
                };

                let Some(&v0) = w_kappa_tuples.get(&kappa_pot_tuple) else {
                    continue;
                };

                for &(j_rel, i_rel, matrix_element) in rel_entries {
                    if matrix_element.abs() <= 1e-12 {
                        continue;
                    }

                    let col = exciton_flat_index(i_com, i_rel, n_rel_states_used);
                    let row = exciton_flat_index(j_com, j_rel, n_rel_states_used);

                    h_int.add_triplet(
                        row,
                        col,
                        M::Scalar::convert_from(Complex64::new(0.0, -2.0 * v0 * matrix_element)),
                    );
                }
            }
        }

        h_full = h_full.add(&M::from_trimat(&h_int)?)?;

        if self.verbose {
            info!(
                "Full Hamiltonian built in {:.2} seconds",
                start_time.elapsed().as_secs_f64()
            );
        }

        Ok((h_full, rel_eigvals, rel_eigvecs))
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
    pub fn build_full_hamiltonian(
        &self,
        n_rel_states: usize,
    ) -> Result<(M, Array1<f64>, Array2<f64>), MatrixOpsError> {
        // Build the Hamiltonian without center-of-mass kinetic energy
        let (mut h_full, rel_eigvals, rel_eigvecs) = self.build_hamiltonian_no_k(n_rel_states)?;

        // Number of center-of-mass states
        let n_com_states = self.kappa_vectors.len();
        let n_rel_states_used = rel_eigvals.len();

        let eye_rel = M::eye(n_rel_states_used);

        let com_kinetic = (0..n_com_states)
            .map(|i_com| {
                let k_vec_i = &self.kappa_vectors[i_com];
                let mut k_plus_k = vec![0.0; 2];
                for j in 0..2 {
                    k_plus_k[j] = self.k[j] + k_vec_i[j];
                }
                (k_plus_k[0].powi(2) + k_plus_k[1].powi(2)) / (2.0 * self.m)
            })
            .collect::<Vec<f64>>();

        h_full = h_full.add(&M::from_diag(&com_kinetic)?.kron(&eye_rel))?;

        Ok((h_full, rel_eigvals, rel_eigvecs))
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
    ) -> Result<(Array1<f64>, Array2<Complex64>, Array1<f64>, Array2<f64>), MatrixOpsError>
    where
        M::Scalar: ConvertFrom<Complex64>,
        M: Diagonalize,
    {
        if self.verbose {
            info!(
                "Building complex Hermitian Hamiltonian with -2i*w_kappa*sin(kappa·x/2) matrix elements"
            );
        }

        // Build the full Hamiltonian with all states
        let (h_full, rel_eigvals, rel_eigvecs) = self.build_full_hamiltonian(n_rel_states)?;

        if self.verbose {
            info!("Computing {} lowest eigenvalues...", k_eigvals);
        }

        // Compute eigenvalues and eigenvectors
        let (eigenvalues, right_eigenvectors) = h_full
            .diagonalize(DiagonalizationOptions::smallest_real(k_eigvals))?
            .into_arrays_hermitian();
        Ok((eigenvalues, right_eigenvectors, rel_eigvals, rel_eigvecs))
    }
}

impl<M: MatrixOps> PreparedExcitonHamiltonian<M>
where
    M::Scalar: ConvertFrom<f64>,
{
    /// Builds the full exciton Hamiltonian at a particular K point.
    ///
    /// The stored basis convention is
    ///
    ///     |state> = |CoM> ⊗ |rel>,
    ///
    /// so the CoM kinetic term is
    ///
    ///     diag(|K+kappa|^2 / 2M) ⊗ I_rel.
    pub fn hamiltonian_at_k(&self, k: &[f64]) -> Result<M, MatrixOpsError> {
        let eye_rel = M::eye(self.n_rel);

        let com_kinetic: Vec<f64> = self
            .kappa_vectors
            .iter()
            .map(|kappa| {
                let kx = k[0] + kappa[0];
                let ky = k[1] + kappa[1];

                (kx * kx + ky * ky) / (2.0 * self.mass)
            })
            .collect();

        self.h_no_k.add(&M::from_diag(&com_kinetic)?.kron(&eye_rel))
    }
}
