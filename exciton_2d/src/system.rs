// src/system.rs
use crate::band_structure::BandStructureCalculator;
use crate::config::ExcitonConfig;
use crate::grid::Grid;
use crate::hamiltonian::{FullHamiltonian, RelativeHamiltonian};
use crate::matrix_elements::MatrixElementCalculator;
use crate::utils::print_memory_usage;
use crate::visualization::{plot_band_structure, plot_rel_eigenstate, plot_spectrum};

use log::{info, warn};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;
use ordered_float::OrderedFloat;

pub struct Exciton2DGridSystem {
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
    pub vector_sum_to_index:  HashMap<(usize, (OrderedFloat<f64>, OrderedFloat<f64>)), usize>,
    pub matrix_elem_calc: MatrixElementCalculator,
    pub rel_ham: RelativeHamiltonian,
    pub full_ham: FullHamiltonian,
    pub band_calc: BandStructureCalculator,
}

impl Exciton2DGridSystem {
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
            fs::create_dir_all(&save_dir).expect("Failed to create save directory");
        }
        
        // Masses for CoM and relative motion
        let m = m_e + m_h;  // Total mass
        let mu = (m_e * m_h) / (m_e + m_h);  // Reduced mass
        
        // Set up the grid
        let grid = Grid::new(grid_size, box_size, verbose);
        
        // Define reciprocal lattice vectors and potential
        let (kappa_vectors, w_kappa, vector_sum_to_index) = 
            Self::setup_reciprocal_lattice(n_kappa, a0, v0, verbose);
        
        // Initialize components
        let matrix_elem_calc = MatrixElementCalculator::new(grid.clone(), save_dir.clone(), recalculate);
        let rel_ham = RelativeHamiltonian::new(grid.clone(), mu, save_dir.clone(), recalculate, batch_size, verbose);
        
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
        
        let band_calc = BandStructureCalculator::new(full_ham.clone(), a0, save_dir.clone(), verbose);
        
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
    
    fn setup_reciprocal_lattice(
        n_kappa: usize,
        a0: f64,
        v0: f64,
        verbose: bool,
    ) -> (Vec<Vec<f64>>, // kappa_vectors
        HashMap<(OrderedFloat<f64>, OrderedFloat<f64>), f64>, // w_kappa
        HashMap<(usize, (OrderedFloat<f64>, OrderedFloat<f64>)), usize> // vector_sum_to_index
    ) 
    {
        // Define reciprocal lattice vectors (for a 2D system)
        let b1 = 2.0 * std::f64::consts::PI / a0;  // First reciprocal lattice vector
        let b2 = 2.0 * std::f64::consts::PI / a0;  // Second reciprocal lattice vector

        // Define range for reciprocal lattice indices
        let n_max = n_kappa as i32 / 2;  // Maximum index for reciprocal lattice vectors
        
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
            if (approx_eq(kx.abs(), b1) && approx_eq(ky.abs(), 0.0)) ||
            (approx_eq(kx.abs(), 0.0) && approx_eq(ky.abs(), b2)) {
                w_kappa.insert((kx, ky), v0 / 2.0);
                if verbose {
                    info!("Added potential term for kappa = [{}, {}] with V = {}", kx, ky, v0/2.0);
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
            kappa_vector_to_index.insert((OrderedFloat::<f64>::from(vec[0]), OrderedFloat::<f64>::from(vec[1])), i);
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
                    if approx_eq(vx.into_inner(), k_vec_sum_x) && 
                    approx_eq(vy.into_inner(), k_vec_sum_y) {
                        vector_sum_to_index.insert((i, (kx, ky)), j);
                        found = true;
                        break;
                    }
                }
                
                if !found && verbose {
                    warn!("No matching vector found for sum: [{}, {}]", k_vec_sum_x, k_vec_sum_y);
                }
            }
        }

        if verbose {
            info!("Lookup table built in {:.2} seconds.", start_time.elapsed().as_secs_f64());
        }
        
        (kappa_vectors, w_kappa, vector_sum_to_index)
    }

    
    pub fn solve_rel_hamiltonian(&self, k_eigvals: usize) -> (Array1<f64>, Array2<f64>, Array1<i32>) {
        self.rel_ham.solve(k_eigvals)
    }
    
    pub fn diagonalize_full_hamiltonian(
        &self,
        n_rel_states: usize,
        k_eigvals: usize,
        select_parity: i32,
    ) -> (Array1<f64>, Vec<Array1<f64>>, Array1<f64>, Array2<f64>, Array1<i32>) {
        self.full_ham.diagonalize_full_hamiltonian(n_rel_states, k_eigvals, select_parity)
    }
    
    pub fn plot_rel_eigenstate(&self, eigvec: &Array1<f64>, title: Option<&str>, save_path: Option<&str>) {
        plot_rel_eigenstate(&self.grid, eigvec, title, save_path);
    }
    
    pub fn plot_spectrum(&self, eigvals: &Array1<f64>, save_path: Option<&str>) {
        plot_spectrum(eigvals, &self.k, save_path);
    }
    
    pub fn calculate_band_structure(
        &self,
        n_points: usize,
        n_rel_states: usize,
        n_bands: usize,
        recalculate_bands: bool,
        select_parity: i32,
    ) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Array1<i32>>) {
        self.band_calc.calculate_band_structure(
            n_points,
            n_rel_states,
            n_bands,
            self.v0,
            recalculate_bands,
            select_parity,
        )
    }
    
    pub fn plot_band_structure(
        &self,
        k_points: &[Vec<f64>],
        eigvals_list: &[Array1<f64>],
        parities_list: &[Array1<i32>],
        save_path: Option<&str>,
    ) {
        plot_band_structure(k_points, eigvals_list, parities_list, self.a0, save_path, false);
    }

    pub fn calculate_n_colored_band_structure(
        &self,
        n_points: usize,
        n_rel_states: usize,
        n_bands: usize,
        recalculate_bands: bool,
        select_parity: i32,
    ) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>) {
        crate::visualization::calculate_n_colored_band_structure(
            &self.full_ham,
            n_points,
            n_rel_states,
            n_bands,
            self.a0,
            self.v0,
            select_parity,
            &self.save_dir,
            recalculate_bands,
        )
    }

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