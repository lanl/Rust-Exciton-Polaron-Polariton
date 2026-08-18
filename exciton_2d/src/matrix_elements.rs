//! ## Matrix Element Convention
//!
//! This module computes the **real** sine integrals:
//! ```text
//! S_ij(κ) = ∫ d²r ψ_i(r) sin(κ·r / 2) ψ_j(r)
//! ```
//!
//! The complex prefactor from the lattice Hamiltonian is applied in
//! `hamiltonian.rs`. The full lattice matrix element is:
//! ```text
//! H_latt = -2i w_κ e^{iκ·X} sin(κ·r / 2)
//! ```
//! so after the CoM plane-wave shift, the relative-coordinate factor is:
//! ```text
//! -2i w_κ S_ij(κ).
//! ```
//!
//! This separation lets the matrix-element calculator work with real numbers,
//! while the Hamiltonian builder handles the complex prefactor and CoM momentum shift.
use crate::grid::Grid;
use crate::lattice::KappaKey;
use crate::utils::fast_matrix_element_calc;
use log::{info, warn};
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Calculator for matrix elements of the periodic potential
///
/// This struct handles the calculation and caching of matrix elements between
/// different eigenstates of the relative coordinate Hamiltonian. It uses
/// precomputation and caching strategies to optimize performance for repeated
/// calculations with the same parameters.
#[derive(Clone)]
pub struct MatrixElementCalculator {
    grid: Grid,
    save_dir: String,
    recalculate: bool,
    matrix_element_cache: Arc<RwLock<HashMap<(usize, usize, KappaKey), f64>>>,
    precomputed: Arc<RwLock<HashMap<String, Array2<f64>>>>,
    cached_eigvecs: Arc<RwLock<Option<Array2<f64>>>>,
    pub verbose: bool,
}

impl MatrixElementCalculator {
    /// Creates a new matrix element calculator
    ///
    /// # Arguments
    /// * `grid` - The spatial grid for discretization
    /// * `save_dir` - Directory to save/load precomputed matrix elements
    /// * `recalculate` - Whether to force recalculation of matrix elements
    ///
    /// # Returns
    /// A new `MatrixElementCalculator` instance
    pub fn new(grid: Grid, save_dir: String, recalculate: bool) -> Self {
        Self {
            grid,
            save_dir,
            recalculate,
            matrix_element_cache: Arc::new(RwLock::new(HashMap::new())),
            precomputed: Arc::new(RwLock::new(HashMap::new())),
            cached_eigvecs: Arc::new(RwLock::new(None)),
            verbose: false,
        }
    }

    /// Precomputes terms needed for potential matrix element calculations
    ///
    /// This method precomputes and caches various terms that are frequently used
    /// in matrix element calculations, such as coordinate grids and sine terms
    /// for each reciprocal lattice vector.
    ///
    /// # Arguments
    /// * `w_kappa` - Map of reciprocal lattice vectors to potential coefficients
    pub fn precompute_potential_terms(&self, w_kappa: &HashMap<KappaKey, f64>, a0: f64) {
        // Precompute x and y grid points for potential calculation
        let mut pre = self.precomputed.write().unwrap();

        // Create x_grid and y_grid
        let mut x_grid = Array2::zeros((self.grid.grid_size, 1));
        let mut y_grid = Array2::zeros((1, self.grid.grid_size));

        for i in 0..self.grid.grid_size {
            x_grid[[i, 0]] = self.grid.x[i];
            y_grid[[0, i]] = self.grid.y[i];
        }

        pre.insert("x_grid".to_string(), x_grid);
        pre.insert("y_grid".to_string(), y_grid);

        // Precompute potential terms for common kappa values
        for &kappa_key in w_kappa.keys() {
            let [kx, ky] = kappa_key.vector(a0);
            let mut sin_term = Array2::zeros((self.grid.grid_size, self.grid.grid_size));

            for iy in 0..self.grid.grid_size {
                for ix in 0..self.grid.grid_size {
                    let x = self.grid.x[ix];
                    let y = self.grid.y[iy];

                    sin_term[[iy, ix]] = ((kx * x + ky * y) / 2.0).sin();
                }
            }

            let key = format!("sin_term_{:.6}_{:.6}", kx, ky);
            pre.insert(key, sin_term);
        }
        drop(pre)
    }

    /// Sets the cached eigenvectors for matrix element calculations
    ///
    /// This method stores the eigenvectors of the relative coordinate Hamiltonian
    /// for use in matrix element calculations.
    ///
    /// # Arguments
    /// * `eigvecs` - Eigenvectors of the relative coordinate Hamiltonian
    pub fn set_cached_eigvecs(&self, eigvecs: Array2<f64>) {
        *self.cached_eigvecs.write().unwrap() = Some(eigvecs);
    }

    /// Calculates or retrieves a cached potential matrix element
    ///
    /// This method computes the matrix element of sin(kappa·r) between eigenstates i and j,
    /// using cached values when available to avoid redundant calculations.
    ///
    /// Note: This returns the REAL sine integral. The -i factor is applied when
    /// constructing the Hamiltonian in hamiltonian.rs.
    ///
    /// # Arguments
    /// * `kappa_pot` - Reciprocal lattice vector
    /// * `i` - Index of the first eigenstate
    /// * `j` - Index of the second eigenstate
    ///
    /// # Returns
    /// The matrix element <i|sin(kappa·r)|j> (real)
    pub fn cached_potential_matrix_element(
        &self,
        kappa_pot: KappaKey,
        i: usize,
        j: usize,
        a0: f64,
    ) -> f64 {
        let cache_key = (i, j, kappa_pot);

        if let Some(&result) = self.matrix_element_cache.read().unwrap().get(&cache_key) {
            return result;
        }

        let eigvecs_guard = self.cached_eigvecs.read().unwrap();
        let eigvecs = eigvecs_guard
            .as_ref()
            .expect("Eigenvectors must be calculated before using cached matrix elements");

        let [kx, ky] = kappa_pot.vector(a0);
        let sin_term_key = format!("sin_term_{:.6}_{:.6}", kx, ky);

        let sin_term = {
            let precomputed = self.precomputed.read().unwrap();

            if let Some(term) = precomputed.get(&sin_term_key) {
                term.clone()
            } else {
                drop(precomputed);

                let mut sin_term = Array2::zeros((self.grid.grid_size, self.grid.grid_size));

                for iy in 0..self.grid.grid_size {
                    for ix in 0..self.grid.grid_size {
                        let x = self.grid.x[ix];
                        let y = self.grid.y[iy];

                        sin_term[[iy, ix]] = ((kx * x + ky * y) / 2.0).sin();
                    }
                }

                let mut precomputed = self.precomputed.write().unwrap();
                precomputed.insert(sin_term_key.clone(), sin_term.clone());

                sin_term
            }
        };

        let psi_i: Array2<f64> = eigvecs
            .column(i)
            .to_shape((self.grid.grid_size, self.grid.grid_size))
            .unwrap()
            .to_owned();

        let psi_j: Array2<f64> = eigvecs
            .column(j)
            .to_shape((self.grid.grid_size, self.grid.grid_size))
            .unwrap()
            .to_owned();

        let result =
            fast_matrix_element_calc(psi_i.view(), psi_j.view(), sin_term.view(), self.grid.dx);

        self.matrix_element_cache
            .write()
            .unwrap()
            .insert(cache_key, result);

        result
    }

    /// Calculates a matrix element directly without caching
    ///
    /// This helper method computes the matrix element of sin(kappa·r) between
    /// eigenstates i and j directly, without using the cache.
    ///
    /// Note: This returns the REAL sine integral. The -i factor is applied when
    /// constructing the Hamiltonian in hamiltonian.rs.
    ///
    /// # Arguments
    /// * `kappa_pot` - Reciprocal lattice vector
    /// * `i` - Index of the first eigenstate
    /// * `j` - Index of the second eigenstate
    /// * `eigvecs` - Eigenvectors of the relative coordinate Hamiltonian
    /// * `x_grid` - Grid of x coordinates
    /// * `y_grid` - Grid of y coordinates
    ///
    /// # Returns
    /// The matrix element <i|sin(kappa·r)|j> (real)
    fn _calculate_matrix_element_directly(
        &self,
        kappa_pot: KappaKey,
        i: usize,
        j: usize,
        a0: f64,
        eigvecs: &Array2<f64>,
    ) -> f64 {
        // Get the eigenvectors for states i and j
        let psi_i: Array2<f64> = eigvecs
            .column(i)
            .to_shape((self.grid.grid_size, self.grid.grid_size))
            .unwrap()
            .to_owned();
        let psi_j = eigvecs
            .column(j)
            .to_shape((self.grid.grid_size, self.grid.grid_size))
            .unwrap()
            .to_owned();

        // Calculate the sine term
        let mut sin_term = Array2::zeros((self.grid.grid_size, self.grid.grid_size));
        let [kx, ky] = kappa_pot.vector(a0);

        for iy in 0..self.grid.grid_size {
            for ix in 0..self.grid.grid_size {
                let x = self.grid.x[ix];
                let y = self.grid.y[iy];

                sin_term[[iy, ix]] = ((kx * x + ky * y) / 2.0).sin();
            }
        }

        fast_matrix_element_calc(psi_i.view(), psi_j.view(), sin_term.view(), self.grid.dx)
    }

    /// Calculates a potential matrix element directly with provided eigenvectors
    ///
    /// This method computes the matrix element of sin(kappa·r) between eigenstates i and j
    /// using the provided eigenvectors, rather than the cached ones.
    ///
    /// Note: This returns the REAL sine integral. The -i factor is applied when
    /// constructing the Hamiltonian in hamiltonian.rs.
    ///
    /// # Arguments
    /// * `kappa_pot` - Reciprocal lattice vector as a slice
    /// * `i` - Index of the first eigenstate
    /// * `j` - Index of the second eigenstate
    /// * `rel_eigvecs` - Eigenvectors of the relative coordinate Hamiltonian
    ///
    /// # Returns
    /// The matrix element <i|sin(kappa·r)|j> (real)
    pub fn get_potential_matrix_element_direct(
        &self,
        kappa_pot: &[f64],
        i: usize,
        j: usize,
        rel_eigvecs: &Array2<f64>,
    ) -> f64 {
        // Direct float-vector path intentionally bypasses the integer-key cache.
        // Prefer the KappaKey-based cached path for lattice-generated vectors.

        // Get the eigenvectors for states i and j
        let psi_i: Array2<f64> = rel_eigvecs
            .column(i)
            .to_shape((self.grid.grid_size, self.grid.grid_size))
            .unwrap()
            .to_owned();
        let psi_j = rel_eigvecs
            .column(j)
            .to_shape((self.grid.grid_size, self.grid.grid_size))
            .unwrap()
            .to_owned();

        // Get precomputed sine term if available
        let precomputed = self.precomputed.read().unwrap();
        let sin_term_key = format!("sin_term_{:.6}_{:.6}", kappa_pot[0], kappa_pot[1]);

        let sin_term = if let Some(term) = precomputed.get(&sin_term_key) {
            term.clone()
        } else {
            let mut sin_term = Array2::zeros((self.grid.grid_size, self.grid.grid_size));

            for iy in 0..self.grid.grid_size {
                for ix in 0..self.grid.grid_size {
                    let x = self.grid.x[ix];
                    let y = self.grid.y[iy];

                    sin_term[[iy, ix]] = ((kappa_pot[0] * x + kappa_pot[1] * y) / 2.0).sin();
                }
            }

            sin_term
        };

        let matrix_element =
            fast_matrix_element_calc(psi_i.view(), psi_j.view(), sin_term.view(), self.grid.dx);

        matrix_element
    }

    /// Calculates all required matrix elements for the full Hamiltonian
    ///
    /// This method computes all matrix elements needed for constructing the full Hamiltonian,
    /// either by loading from disk or calculating them directly. Only non-zero matrix elements
    /// are stored to save memory.
    ///
    /// Note: This returns REAL sine integrals. The -i factor is applied when
    /// constructing the Hamiltonian in hamiltonian.rs.
    ///
    /// # Arguments
    /// * `w_kappa_tuples` - Map of reciprocal lattice vectors to potential coefficients
    /// * `n_rel_states` - Number of relative coordinate states to include
    ///
    /// # Returns
    /// A map from (i, j, kappa) to the corresponding matrix element <i|sin(kappa·r)|j> (real)
    pub fn calculate_matrix_elements(
        &self,
        w_kappa_tuples: &HashMap<KappaKey, f64>,
        n_rel_states: usize,
        a0: f64,
    ) -> HashMap<(usize, usize, KappaKey), f64> {
        let matrix_elements_filename = format!(
            "{}/matrix_elements_grid{}_box{}_nrel{}.json",
            self.save_dir, self.grid.grid_size, self.grid.box_size, n_rel_states
        );

        if Path::new(&matrix_elements_filename).exists() && !self.recalculate && self.verbose {
            info!("Cache exists but loading is not implemented; recomputing...");
        }

        if self.verbose {
            info!("Pre-calculating potential matrix elements...");
        }

        let mut v_rel = HashMap::new();

        for (&kappa_pot_tuple, _) in w_kappa_tuples {
            for i in 0..n_rel_states {
                for j in i..n_rel_states {
                    let matrix_element =
                        self.cached_potential_matrix_element(kappa_pot_tuple, i, j, a0);

                    if matrix_element.abs() > 1e-12 {
                        v_rel.insert((i, j, kappa_pot_tuple), matrix_element);

                        if i != j {
                            v_rel.insert((j, i, kappa_pot_tuple), matrix_element);
                        }
                    }
                }
            }
        }

        if self.verbose {
            info!(
                "Computed {} non-zero matrix elements for {} non-zero kappa values",
                v_rel.len(),
                w_kappa_tuples.len()
            );

            if !v_rel.is_empty() {
                info!("Sample matrix elements:");
                for (&(i_rel, j_rel, kappa_pot), &val) in v_rel.iter().take(5) {
                    info!("  <{}|sin({:?}·r/2)|{}> = {}", i_rel, kappa_pot, j_rel, val);
                }
            } else {
                warn!("WARNING: No non-zero matrix elements found!");
            }
        }

        v_rel
    }
}
