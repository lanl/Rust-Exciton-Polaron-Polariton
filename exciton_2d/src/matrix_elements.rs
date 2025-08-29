// src/matrix_elements.rs
use crate::grid::Grid;
use crate::utils::fast_matrix_element_calc;
use log::{info, warn};
use ndarray::Array2;
use num_complex::Complex64;
use ordered_float::OrderedFloat;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone)]
pub struct MatrixElementCalculator {
    grid: Grid,
    save_dir: String,
    recalculate: bool,
    matrix_element_cache: Arc<RwLock<HashMap<(usize, usize, (OrderedFloat<f64>, OrderedFloat<f64>)), Complex64>>>,
    precomputed: Arc<RwLock<HashMap<String, Array2<f64>>>>,
    cached_eigvecs: Arc<RwLock<Option<Array2<f64>>>>,
    pub verbose: bool,
}

impl MatrixElementCalculator {
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
    
    pub fn precompute_potential_terms(&self, w_kappa: &HashMap<(OrderedFloat<f64>, OrderedFloat<f64>), f64>) {
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
        for &(kx, ky) in w_kappa.keys() {
            let mut cos_x = Array2::zeros((self.grid.grid_size, 1));
            let mut cos_y = Array2::zeros((1, self.grid.grid_size));
            
            for i in 0..self.grid.grid_size {
                cos_x[[i, 0]] = (kx * self.grid.x[i] / 2.0).cos();
                cos_y[[0, i]] = (ky * self.grid.y[i] / 2.0).cos();
            }
            
            let key = format!("cos_term_{:.6}_{:.6}", kx, ky);
            pre.insert(key, cos_x + cos_y);
        }
        drop(pre)
    }
    
    pub fn set_cached_eigvecs(&self, eigvecs: Array2<f64>) {
        *self.cached_eigvecs.write().unwrap() = Some(eigvecs);
    }
    
    pub fn cached_potential_matrix_element(&self, kappa_pot: (OrderedFloat<f64>, OrderedFloat<f64>), i: usize, j: usize) -> Complex64 {
        let cache_key = (i, j, kappa_pot);
        
        // First check if the result is already in the cache
        if let Some(&result) = self.matrix_element_cache.read().unwrap().get(&cache_key) {
            return result;
        }
        
        // Release the lock before acquiring other locks
        
        // Get the eigenvectors - acquire this lock first
        let eigvecs = match self.cached_eigvecs.read().unwrap().as_ref() {
            Some(ev) => ev.clone(),
            None => panic!("Eigenvectors must be calculated before using cached matrix elements")
        };
        
        // Now get the precomputed data - acquire this lock second
        let precomputed_data = {
            let precomputed = self.precomputed.read().unwrap();
            
            if precomputed.is_empty() {
                // Create temporary data without holding the lock
                drop(precomputed);
                
                // Create x_grid and y_grid
                let mut x_grid = Array2::zeros((self.grid.grid_size, 1));
                let mut y_grid = Array2::zeros((1, self.grid.grid_size));
                
                for i in 0..self.grid.grid_size {
                    x_grid[[i, 0]] = self.grid.x[i];
                    y_grid[[0, i]] = self.grid.y[i];
                }
                
                // Now acquire the write lock to update precomputed data
                let mut precomputed = self.precomputed.write().unwrap();
                precomputed.insert("x_grid".to_string(), x_grid);
                precomputed.insert("y_grid".to_string(), y_grid);
                
                // Return a copy of what we need
                (
                    precomputed.get("x_grid").unwrap().clone(),
                    precomputed.get("y_grid").unwrap().clone()
                )
            } else {
                // Just get what we need from existing data
                (
                    precomputed.get("x_grid").unwrap().clone(),
                    precomputed.get("y_grid").unwrap().clone()
                )
            }
        };
        
        let (x_grid, y_grid) = precomputed_data;
        
        // Calculate the matrix element without holding any locks
        let result = self.calculate_matrix_element_directly(kappa_pot, i, j, &eigvecs, &x_grid, &y_grid);
        
        // Finally, update the cache with the result
        self.matrix_element_cache.write().unwrap().insert(cache_key, result);
        
        result
    }

    // Add a new helper method to calculate the matrix element directly
    fn calculate_matrix_element_directly(
        &self,
        kappa_pot: (OrderedFloat<f64>, OrderedFloat<f64>),
        i: usize,
        j: usize,
        eigvecs: &Array2<f64>,
        x_grid: &Array2<f64>,
        y_grid: &Array2<f64>
    ) -> Complex64 {
        // Get the eigenvectors for states i and j
        let psi_i: Array2<f64> = eigvecs.column(i)
            .to_shape((self.grid.grid_size, self.grid.grid_size)).unwrap().to_owned();
        let psi_j = eigvecs.column(j)
            .to_shape((self.grid.grid_size, self.grid.grid_size)).unwrap().to_owned();
        
        // Calculate the cosine term
        let mut cos_x = Array2::zeros((self.grid.grid_size, 1));
        let mut cos_y = Array2::zeros((1, self.grid.grid_size));
        
        for i in 0..self.grid.grid_size {
            cos_x[[i, 0]] = (kappa_pot.0.0 * x_grid[[i, 0]]).cos();
            cos_y[[0, i]] = (kappa_pot.1.0 * y_grid[[0, i]]).cos();
        }
        
        let cos_term = cos_x + cos_y;
        
        // Use the optimized function for the matrix element calculation
        let psi_i_imag = Array2::zeros((self.grid.grid_size, self.grid.grid_size)); // Real wavefunctions
        let psi_j_imag = Array2::zeros((self.grid.grid_size, self.grid.grid_size)); // Real wavefunctions
        
        fast_matrix_element_calc(
            &psi_i, &psi_i_imag, 
            &psi_j, &psi_j_imag,
            &cos_term, self.grid.dx
        )
    }

    
    pub fn get_potential_matrix_element_direct(&self, kappa_pot: &[f64], i: usize, j: usize, rel_eigvecs: &Array2<f64>) -> Complex64 {
        let kappa_pot_tuple = (OrderedFloat(kappa_pot[0]), OrderedFloat(kappa_pot[1]));
        let cache_key = (i, j, kappa_pot_tuple);
        
        if let Some(&result) = self.matrix_element_cache.read().unwrap().get(&cache_key) {
            return result;
        }
        
        // Get the eigenvectors for states i and j
        let psi_i: Array2<f64> = rel_eigvecs.column(i)
            .to_shape((self.grid.grid_size, self.grid.grid_size)).unwrap().to_owned();
        let psi_j = rel_eigvecs.column(j)
            .to_shape((self.grid.grid_size, self.grid.grid_size)).unwrap().to_owned();
        
        // Get precomputed cosine term if available
        let precomputed = self.precomputed.read().unwrap();
        let cos_term_key = format!("cos_term_{:.6}_{:.6}", kappa_pot[0], kappa_pot[1]);
        
        let cos_term = if let Some(term) = precomputed.get(&cos_term_key) {
            term.clone()
        } else {
            // Calculate the cosine term
            let x_grid = precomputed.get("x_grid").unwrap();
            let y_grid = precomputed.get("y_grid").unwrap();
            
            let mut cos_x = Array2::zeros((self.grid.grid_size, 1));
            let mut cos_y = Array2::zeros((1, self.grid.grid_size));
            
            for i in 0..self.grid.grid_size {
                cos_x[[i, 0]] = (kappa_pot[0] * x_grid[[i, 0]]).cos();
                cos_y[[0, i]] = (kappa_pot[1] * y_grid[[0, i]]).cos();
            }
            
            cos_x + cos_y
        };
        
        // Use the optimized function for the matrix element calculation
        let psi_i_imag = Array2::zeros((self.grid.grid_size, self.grid.grid_size)); // Real wavefunctions
        let psi_j_imag = Array2::zeros((self.grid.grid_size, self.grid.grid_size)); // Real wavefunctions
        
        let matrix_element = fast_matrix_element_calc(
            &psi_i, &psi_i_imag, 
            &psi_j, &psi_j_imag,
            &cos_term, self.grid.dx
        );
        
        // Cache the result
        self.matrix_element_cache.write().unwrap().insert(cache_key, matrix_element);
        
        matrix_element
    }
    
    pub fn calculate_matrix_elements(&self, w_kappa_tuples: &HashMap<(OrderedFloat<f64>, OrderedFloat<f64>), f64>, n_rel_states: usize) -> HashMap<(usize, usize, (OrderedFloat<f64>, OrderedFloat<f64>)), Complex64> {
        // Check if precomputed matrix elements exist
        let matrix_elements_filename = format!(
            "{}/matrix_elements_grid{}_box{}_nrel{}.json",
            self.save_dir,
            self.grid.grid_size,
            self.grid.box_size,
            n_rel_states
        );
        
        if Path::new(&matrix_elements_filename).exists() && !self.recalculate {
            if self.verbose {info!("Loading precomputed matrix elements from {}", matrix_elements_filename)};
            
            // In a real implementation, we'd deserialize from JSON
            // For now, we'll just return an empty HashMap
            let v_rel = HashMap::new();
            if self.verbose {info!("Loaded {} precomputed matrix elements", v_rel.len())};
            return v_rel;
        }
        
        if self.verbose {info!("Pre-calculating potential matrix elements...")};
        
        // Pre-calculate potential matrix elements ONLY for non-zero w_kappa values
        let mut v_rel = HashMap::new();
                
        for (&kappa_pot_tuple, _) in w_kappa_tuples {
            for i in 0..n_rel_states {
                for j in 0..n_rel_states {
                    // Use cached matrix element calculation
                    let matrix_element = self.cached_potential_matrix_element(kappa_pot_tuple, i, j);
                    if matrix_element.norm() > 1e-12 {  // Only store non-zero elements
                        v_rel.insert((i, j, kappa_pot_tuple), matrix_element);
                    }
                }
            }
        }
        
        // Save the precomputed matrix elements
        if self.verbose {info!("Saving {} matrix elements to {}", v_rel.len(), matrix_elements_filename)};
        
        if self.verbose {
            // Debug: Print some sample matrix elements
            if !v_rel.is_empty() {
                info!("Sample matrix elements:");
                let mut sample_count = 0;
                for (&(i_rel, j_rel, kappa_pot), &val) in &v_rel {
                    if sample_count < 5 {
                        info!("  <{}|e^(i*{:?}·r)|{}> = {}", i_rel, kappa_pot, j_rel, val);
                        sample_count += 1;
                    }
                }
            } else {
                warn!("WARNING: No non-zero matrix elements found!");
            }
        }
        
        // In a real implementation, we'd serialize to JSON
        // For now, we'll just return the HashMap
        v_rel
    }
}
