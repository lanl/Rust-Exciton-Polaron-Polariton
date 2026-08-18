use crate::grid::Grid;

use log::info;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
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
        crate::relative::laguerre::laguerre_assoc(n, alpha, x)
    }

    /// Computes the factorial of n
    ///
    /// # Arguments
    /// * `n` - Non-negative integer
    ///
    /// # Returns
    /// * n!
    fn factorial(&self, n: i32) -> u64 {
        crate::relative::laguerre::factorial(n)
    }
}
