//! Main system module for exciton simulations
//!
//! This module provides the main system class that integrates all components
//! for simulating excitons in 2D materials.

use crate::band_structure::BandStructureCalculator;
use crate::band_structure::calculate_n_colored_band_structure;
use crate::config::ExcitonConfig;
use crate::grid::Grid;
use crate::hamiltonian::{FullHamiltonian, RelativeHamiltonian};
use crate::matrix_elements::MatrixElementCalculator;
use crate::utils::print_memory_usage;
use crate::visualization::{plot_band_structure, plot_rel_eigenstate, plot_spectrum};

use log::{info, warn};
use matrix_ops::MatrixOps;
use matrix_ops::{ConvertFrom, Diagonalize, MatrixOpsError};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Main system class for exciton simulations in 2D materials
///
/// This struct serves as the central interface for simulating excitons in 2D materials
/// with periodic potentials. It integrates all components needed for the simulation:
/// - Grid for spatial discretization
/// - Hamiltonian for energy calculations
/// - Matrix element calculator for potential terms
/// - Band structure calculator for dispersion relations
///
/// It provides methods for solving the exciton Hamiltonian, calculating band structures,
/// and visualizing results.
pub struct Exciton2DGridSystem<M: MatrixOps> {
    pub n_kappa: usize,
    pub a0: f64,
    pub v0: f64,
    pub k: Vec<f64>,
    pub m_e: f64,
    pub m_h: f64,
    pub verbose: bool,
    pub grid_size: usize,
    pub box_size: f64,
    pub save_dir: String,
    pub recalculate: bool,
    pub n_jobs: i32,
    pub use_gpu: bool,
    pub batch_size: usize,
    pub m: f64,
    pub mu: f64,
    pub grid: Grid,
    pub kappa_vectors: Vec<Vec<f64>>,
    pub w_kappa: HashMap<(OrderedFloat<f64>, OrderedFloat<f64>), f64>,
    pub vector_sum_to_index: HashMap<(usize, (OrderedFloat<f64>, OrderedFloat<f64>)), usize>,
    pub matrix_elem_calc: MatrixElementCalculator,
    pub rel_ham: RelativeHamiltonian,
    pub full_ham: FullHamiltonian<M>,
    pub band_calc: BandStructureCalculator<M>,
}

impl<M: MatrixOps> Exciton2DGridSystem<M>
where
    M::Scalar: ConvertFrom<Complex64>,
    M: Diagonalize,
{
    /// Creates a new exciton system with the specified parameters
    ///
    /// # Arguments
    /// * `n_kappa` - Number of reciprocal lattice vectors in each dimension
    /// * `a0` - Lattice constant
    /// * `v0` - Potential strength
    /// * `k` - Wave vector
    /// * `m_e` - Electron effective mass
    /// * `m_h` - Hole effective mass
    /// * `verbose` - Whether to print detailed information
    /// * `grid_size` - Number of grid points in each dimension
    /// * `box_size` - Physical size of the simulation box
    /// * `save_dir` - Directory to save/load data
    /// * `recalculate` - Whether to force recalculation of results
    /// * `n_jobs` - Number of parallel jobs (-1 for all cores)
    /// * `use_gpu` - Whether to use GPU acceleration
    /// * `batch_size` - Batch size for memory-intensive operations
    pub fn new(
        n_kappa: usize,
        a0: f64,
        v0: f64,
        k: Vec<f64>,
        m_e: f64,
        m_h: f64,
        verbose: bool,
        grid_size: usize,
        box_size: f64,
        save_dir: String,
        recalculate: bool,
        n_jobs: i32,
        use_gpu: bool,
        batch_size: usize,
    ) -> Self {
        // Create save directory if it doesn't exist
        if !Path::new(&save_dir).exists() {
            match fs::create_dir_all(&save_dir) {
                Ok(_) => {
                    if verbose {
                        info!("Created save directory: {}", save_dir);
                    }
                }
                Err(e) => {
                    warn!("Failed to create save directory {}: {}", save_dir, e);
                    // Continue anyway, operations that need the directory will fail later
                }
            }
        }

        // Masses for CoM and relative motion
        let m = m_e + m_h; // Total mass
        let mu = (m_e * m_h) / (m_e + m_h); // Reduced mass

        // Set up the grid
        let grid = Grid::new(grid_size, box_size, verbose);

        // Define reciprocal lattice vectors and potential
        let (kappa_vectors, w_kappa, vector_sum_to_index) =
            Self::setup_reciprocal_lattice(n_kappa, a0, v0, verbose);

        // Initialize components
        let matrix_elem_calc =
            MatrixElementCalculator::new(grid.clone(), save_dir.clone(), recalculate);
        let rel_ham = RelativeHamiltonian::new(
            grid.clone(),
            mu,
            save_dir.clone(),
            recalculate,
            batch_size,
            verbose,
        );

        let full_ham = FullHamiltonian::new(
            rel_ham.clone(),
            matrix_elem_calc.clone(),
            kappa_vectors.clone(),
            vector_sum_to_index.clone(),
            w_kappa.clone(),
            m,
            k.clone(),
            verbose,
        );

        let band_calc =
            BandStructureCalculator::new(full_ham.clone(), a0, save_dir.clone(), verbose);

        // Precompute potential terms
        matrix_elem_calc.precompute_potential_terms(&w_kappa);

        // Check memory usage
        if verbose {
            print_memory_usage("After initialization");
        }

        Self {
            n_kappa,
            a0,
            v0,
            k,
            m_e,
            m_h,
            verbose,
            grid_size,
            box_size,
            save_dir,
            recalculate,
            n_jobs,
            use_gpu,
            batch_size,
            m,
            mu,
            grid,
            kappa_vectors,
            w_kappa,
            vector_sum_to_index,
            matrix_elem_calc,
            rel_ham,
            full_ham,
            band_calc,
        }
    }

    /// Creates a new exciton system from a configuration object
    ///
    /// # Arguments
    /// * `config` - Configuration object with all parameters
    pub fn from_config(config: ExcitonConfig) -> Self {
        Self::new(
            config.n_kappa,
            config.a0,
            config.v0,
            config.k,
            config.m_e,
            config.m_h,
            config.verbose,
            config.grid_size,
            config.box_size,
            config.save_dir.to_string_lossy().to_string(),
            config.recalculate,
            config.n_jobs,
            config.use_gpu,
            config.batch_size,
        )
    }

    /// Sets up the reciprocal lattice vectors and potential terms
    ///
    /// This method constructs the reciprocal lattice vectors for a 2D square lattice
    /// and computes the Fourier coefficients of the periodic potential. It also builds
    /// a lookup table for efficient vector sum calculations.
    ///
    /// The potential is assumed to have the form V(r) = v0 * (cos(2π*x/a0) + cos(2π*y/a0)),
    /// which corresponds to a square lattice with lattice constant a0.
    ///
    /// # Arguments
    /// * `n_kappa` - Number of reciprocal lattice vectors in each dimension
    /// * `a0` - Lattice constant
    /// * `v0` - Potential strength
    /// * `verbose` - Whether to print detailed information
    ///
    /// # Returns
    /// * Tuple containing:
    ///   - Vector of reciprocal lattice vectors
    ///   - HashMap mapping vectors to potential coefficients
    ///   - HashMap for efficient vector sum lookups
    fn setup_reciprocal_lattice(
        n_kappa: usize,
        a0: f64,
        v0: f64,
        verbose: bool,
    ) -> (
        Vec<Vec<f64>>,                                                   // kappa_vectors
        HashMap<(OrderedFloat<f64>, OrderedFloat<f64>), f64>,            // w_kappa
        HashMap<(usize, (OrderedFloat<f64>, OrderedFloat<f64>)), usize>, // vector_sum_to_index
    ) {
        // Define reciprocal lattice vectors (for a 2D system)
        let b1 = 2.0 * std::f64::consts::PI / a0; // First reciprocal lattice vector
        let b2 = 2.0 * std::f64::consts::PI / a0; // Second reciprocal lattice vector

        // Define range for reciprocal lattice indices
        let n_max = n_kappa as i32 / 2; // Maximum index for reciprocal lattice vectors

        // Calculate all reciprocal lattice vectors κ = n1*b1 + n2*b2
        let mut kappa_vectors = Vec::new();
        for n1 in -n_max..=n_max {
            for n2 in -n_max..=n_max {
                kappa_vectors.push(vec![n1 as f64 * b1, n2 as f64 * b2]);
            }
        }

        // Fourier coefficients for the periodic potential
        let mut w_kappa = HashMap::new();

        // Define a relative tolerance for floating-point comparisons
        const REL_TOL: f64 = 1e-9;
        const ABS_TOL: f64 = 1e-12;

        // Helper function for approximate floating-point equality
        let approx_eq = |a: f64, b: f64| -> bool {
            let abs_diff = (a - b).abs();
            abs_diff < ABS_TOL || abs_diff / a.abs().max(b.abs()) < REL_TOL
        };

        for vec in &kappa_vectors {
            let kx = OrderedFloat::<f64>::from(vec[0]);
            let ky = OrderedFloat::<f64>::from(vec[1]);

            // Check if this is a first-order reciprocal lattice vector using proper comparison
            if (approx_eq(kx.abs(), b1) && approx_eq(ky.abs(), 0.0))
                || (approx_eq(kx.abs(), 0.0) && approx_eq(ky.abs(), b2))
            {
                w_kappa.insert((kx, ky), v0 / 2.0);
                if verbose {
                    info!(
                        "Added potential term for kappa = [{}, {}] with V = {}",
                        kx,
                        ky,
                        v0 / 2.0
                    );
                }
            }
        }

        // Precompute a lookup table for vector sums to indices
        if verbose {
            info!("Building vector sum lookup table...");
        }

        let start_time = Instant::now();

        // Create a dictionary mapping vector tuples to their indices for fast lookup
        let mut kappa_vector_to_index = HashMap::new();
        for (i, vec) in kappa_vectors.iter().enumerate() {
            kappa_vector_to_index.insert(
                (
                    OrderedFloat::<f64>::from(vec[0]),
                    OrderedFloat::<f64>::from(vec[1]),
                ),
                i,
            );
        }

        // More efficient lookup table construction with proper floating-point comparisons
        let mut vector_sum_to_index = HashMap::new();

        for (i, k_vec_i) in kappa_vectors.iter().enumerate() {
            for (&(kx, ky), _) in &w_kappa {
                // Calculate the sum vector
                let k_vec_sum_x = k_vec_i[0] + **OrderedFloat(kx);
                let k_vec_sum_y = k_vec_i[1] + **OrderedFloat(ky);

                // Find the matching vector using proper tolerance-based comparison
                let mut found = false;
                for (&(vx, vy), &j) in &kappa_vector_to_index {
                    if approx_eq(vx.into_inner(), k_vec_sum_x)
                        && approx_eq(vy.into_inner(), k_vec_sum_y)
                    {
                        vector_sum_to_index.insert((i, (kx, ky)), j);
                        found = true;
                        break;
                    }
                }

                if !found && verbose {
                    warn!(
                        "No matching vector found for sum: [{}, {}]",
                        k_vec_sum_x, k_vec_sum_y
                    );
                }
            }
        }

        if verbose {
            info!(
                "Lookup table built in {:.2} seconds.",
                start_time.elapsed().as_secs_f64()
            );
        }

        (kappa_vectors, w_kappa, vector_sum_to_index)
    }

    /// Solves the relative coordinate Hamiltonian
    ///
    /// # Arguments
    /// * `k_eigvals` - Number of eigenvalues to compute
    ///
    /// # Returns
    /// * Tuple containing eigenvalues, eigenvectors, and parity values
    pub fn solve_rel_hamiltonian(
        &self,
        k_eigvals: usize,
    ) -> (Array1<f64>, Array2<f64>, Array1<i32>) {
        self.rel_ham.solve(k_eigvals)
    }

    /// Diagonalizes the full Hamiltonian including center-of-mass motion
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
        self.full_ham
            .diagonalize_full_hamiltonian(n_rel_states, k_eigvals)
    }

    /// Plots a relative coordinate eigenstate
    ///
    /// # Arguments
    /// * `eigvec` - Eigenstate vector to plot
    /// * `title` - Optional title for the plot
    /// * `save_path` - Optional path to save the plot
    pub fn plot_rel_eigenstate(
        &self,
        eigvec: &Array1<f64>,
        title: Option<&str>,
        save_path: Option<&str>,
    ) {
        plot_rel_eigenstate(&self.grid, eigvec, title, save_path);
    }

    /// Plots the energy spectrum
    ///
    /// # Arguments
    /// * `eigvals` - Eigenvalues to plot
    /// * `save_path` - Optional path to save the plot
    pub fn plot_spectrum(&self, eigvals: &Array1<f64>, save_path: Option<&str>) {
        plot_spectrum(eigvals, &self.k, save_path);
    }

    /// Calculates the band structure along high-symmetry paths
    ///
    /// Note: With sine matrix elements, parity is not conserved. All states are
    /// included in the calculation.
    ///
    /// # Arguments
    /// * `n_points` - Number of k-points along each path segment
    /// * `n_rel_states` - Number of relative coordinate states to include
    /// * `n_bands` - Number of bands to calculate
    /// * `recalculate_bands` - Whether to force recalculation
    ///
    /// # Returns
    /// * Tuple containing k-points, eigenvalues, and parity values (of unperturbed states)
    pub fn calculate_band_structure(
        &self,
        n_points: usize,
        n_rel_states: usize,
        n_bands: usize,
        recalculate_bands: bool,
    ) -> (Vec<Vec<f64>>, Vec<Array1<f64>>) {
        self.band_calc.calculate_band_structure(
            n_points,
            n_rel_states,
            n_bands,
            self.v0,
            recalculate_bands,
        )
    }

    /// Plots the band structure
    ///
    /// # Arguments
    /// * `k_points` - k-points along the path
    /// * `eigvals_list` - List of eigenvalues at each k-point
    /// * `parities_list` - List of parity values at each k-point
    /// * `save_path` - Optional path to save the plot
    pub fn plot_band_structure(
        &self,
        k_points: &[Vec<f64>],
        eigvals_list: &[Array1<f64>],
        save_path: Option<&str>,
    ) {
        plot_band_structure(k_points, eigvals_list, self.a0, save_path, false);
    }

    /// Calculates the band structure with n-expectation values
    ///
    /// This method computes the band structure along high-symmetry paths in the
    /// Brillouin zone, including the expectation value of the principal quantum number
    /// for each eigenstate. These expectation values indicate which relative coordinate
    /// state contributes most to each band, providing insight into the character of the bands.
    ///
    /// Note: With sine matrix elements, parity is not conserved. All states are
    /// included in the calculation.
    ///
    /// # Arguments
    /// * `n_points` - Number of k-points along each path segment
    /// * `n_rel_states` - Number of relative coordinate states to include
    /// * `n_bands` - Number of bands to calculate
    /// * `recalculate_bands` - Whether to force recalculation
    ///
    /// # Returns
    /// * Tuple containing k-points, eigenvalues, and n-expectation values
    pub fn calculate_n_colored_band_structure(
        &self,
        n_points: usize,
        n_rel_states: usize,
        n_bands: usize,
        recalculate_bands: bool,
    ) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>) {
        calculate_n_colored_band_structure(
            &self.full_ham,
            n_points,
            n_rel_states,
            n_bands,
            self.a0,
            self.v0,
            &self.save_dir,
            recalculate_bands,
        )
        .unwrap()
    }

    /// Plots the band structure with n-expectation values as colors
    ///
    /// This method creates a high-quality plot of the exciton band structure with bands
    /// colored according to their n-expectation values. This visualization helps identify
    /// the character of each band and how it changes across the Brillouin zone.
    ///
    /// The plot shows the energy bands along the path Γ→X→M→Γ in the Brillouin zone,
    /// with a color scale indicating the n-expectation value of each state.
    ///
    /// # Arguments
    /// * `k_points` - k-points along the path
    /// * `eigvals_list` - List of eigenvalues at each k-point
    /// * `n_expectation_list` - List of n-expectation values at each k-point
    /// * `n_bands` - Number of bands to plot
    /// * `save_path` - Path to save the plot
    pub fn plot_n_colored_band_structure(
        &self,
        k_points: &Vec<Vec<f64>>,
        eigvals_list: &Vec<Array1<f64>>,
        n_expectation_list: &Vec<Vec<f64>>,
        n_bands: usize,
        save_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        crate::visualization::plot_n_colored_band_structure(
            k_points,
            eigvals_list,
            n_expectation_list,
            n_bands,
            self.a0,
            save_path,
            self.v0,
        )
    }
}
