// src/band_structure.rs
use crate::hamiltonian::FullHamiltonian;
use crate::utils::apply_symmetry;
use log::info;
use ndarray::Array1;
use core::panic;
use std::collections::HashMap;
use std::path::Path;
use rayon::{prelude::*, ThreadPoolBuilder};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;

pub struct BandStructureCalculator {
    pub full_ham: FullHamiltonian,
    pub a0: f64,
    pub save_dir: String,
    pub verbose: bool,
}

impl BandStructureCalculator {
    pub fn new(full_ham: FullHamiltonian, a0: f64, save_dir: String, verbose: bool) -> Self {
        Self {
            full_ham,
            a0,
            save_dir,
            verbose,
        }
    }
    
    pub fn get_band_structure_filename(
        &self,
        n_points: usize,
        n_rel_states: usize,
        n_bands: usize,
        v0: f64,
        select_parity: i32,
    ) -> String {
        format!(
            "{}/band_a0{}_V0{}_grid{}_box{}_nrel{}_nbands{}_npoints{}_par{}.json",
            self.save_dir,
            self.a0,
            v0,
            self.full_ham.rel_ham.grid.grid_size,
            self.full_ham.rel_ham.grid.box_size,
            n_rel_states,
            n_bands,
            n_points,
            select_parity
        )
    }
    
    pub fn calculate_at_k_point(
        &self,
        k_point: &[f64],
        n_rel_states: usize,
        n_bands: usize,
        select_parity: i32,
    ) -> (Array1<f64>, Vec<Array1<f64>>,Array1<i32>) {
        // Store original K value
        let original_k = self.full_ham.k.clone();
        
        // Set the new K value
        let mut full_ham = self.full_ham.clone();
        full_ham.k = k_point.to_vec();
        
        // Diagonalize the Hamiltonian
        let (eigvals, eigvecs, _, _, parities) = full_ham.diagonalize_full_hamiltonian(
            n_rel_states,
            n_bands,
            select_parity,
        );
        
        // Restore original K value
        let mut full_ham = self.full_ham.clone();
        full_ham.k = original_k;
        
        (eigvals, eigvecs, parities)
    }
    
    pub fn calculate_band_structure(
        &self,
        n_points: usize,
        n_rel_states: usize,
        n_bands: usize,
        v0: f64,
        recalculate_bands: bool,
        select_parity: i32,
    ) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Array1<i32>>) {
        // Check if we should try to load from file
        let filename = self.get_band_structure_filename(n_points, n_rel_states, n_bands, v0, select_parity);
        
        if !recalculate_bands && Path::new(&filename).exists() {
            if self.verbose {
                info!("Loading band structure data from {}", filename);
            }
            
            panic!("Loading data not implemented yet");
        }
        
        // Define high-symmetry points in the Brillouin zone
        // For a 2D square lattice:
        let b1 = 2.0 * std::f64::consts::PI / self.a0;  // Reciprocal lattice constant
        
        // High-symmetry points
        let gamma = vec![0.0, 0.0];           // Γ point (center of BZ)
        let m = vec![b1/2.0, b1/2.0];         // M point (corner of BZ)
        let k = vec![b1/2.0, 0.0];            // K point (edge midpoint of BZ)
        
        // Create segments of the path
        let points_per_segment = n_points / 3;
        
        // Γ → K
        let mut segment1 = Vec::new();
        for t in 0..points_per_segment {
            let t_frac = t as f64 / (points_per_segment - 1) as f64;
            let x = gamma[0] + t_frac * (k[0] - gamma[0]);
            let y = gamma[1] + t_frac * (k[1] - gamma[1]);
            segment1.push(vec![x, y]);
        }

        // K → M
        let mut segment2 = Vec::new();
        for t in 0..points_per_segment {
            let t_frac = t as f64 / (points_per_segment - 1) as f64;
            let x = k[0] + t_frac * (m[0] - k[0]);
            let y = k[1] + t_frac * (m[1] - k[1]);
            segment2.push(vec![x, y]);
        }
        
        // M → Γ
        let mut segment3 = Vec::new();
        for t in 0..points_per_segment {
            let t_frac = t as f64 / (points_per_segment - 1) as f64;
            let x = m[0] + t_frac * (gamma[0] - m[0]);
            let y = m[1] + t_frac * (gamma[1] - m[1]);
            segment3.push(vec![x, y]);
        }
        
        // Combine all segments
        let mut k_points = Vec::new();
        k_points.extend(segment1);
        k_points.extend(segment2);
        k_points.extend(segment3);
        
        // Use symmetry to reduce calculations
        // For a square lattice, we have 8 symmetry operations
        let symmetry_operations: Vec<usize> = (0..8).collect();
        
        // Identify unique K points that can't be obtained by symmetry
        if self.verbose {
            info!("Identifying unique K points using symmetry...");
        }
        
        let mut unique_k_points: Vec<Vec<f64>> = Vec::new();
        let mut k_point_to_unique = HashMap::new();
        
        for (i, k_point) in k_points.iter().enumerate() {
            // Check if this point can be obtained by symmetry from points we've already processed
            let mut found = false;
            for (j, uk_point) in unique_k_points.iter().enumerate() {
                for &sym_op in &symmetry_operations {
                    let transformed = apply_symmetry(uk_point, sym_op);
                    if (k_point[0] - transformed[0]).abs() < 1e-10 && (k_point[1] - transformed[1]).abs() < 1e-10 {
                        k_point_to_unique.insert(i, (j, sym_op));
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            
            if !found {
                k_point_to_unique.insert(i, (unique_k_points.len(), 0)); // 0 = identity operation
                unique_k_points.push(k_point.clone());
            }
        }
        
        if self.verbose {
            info!("Reduced from {} to {} unique K points using symmetry",
                  k_points.len(), unique_k_points.len());
        }
        
        // Calculate eigenvalues for unique K points
        if self.verbose {
            info!("Calculating band structure at {} unique K points...", unique_k_points.len());
        }
        
        let start_time = Instant::now();

        // Create a progress bar
        let progress_bar = ProgressBar::new(unique_k_points.len() as u64);
        progress_bar.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap());

        let num_threads = 2;
        
        // Create a custom thread pool for this calculation
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)  // Adjust as needed
            .build()
            .unwrap();
        
        // let _limiter = ThreadLimiter::new("OPENBLAS_NUM_THREADS", "1");

        let (unique_eigvals, unique_parities): (Vec<_>, Vec<_>) = pool.install(|| {
            // Parallelize the computation
            unique_k_points.par_iter()
                .map(|k_point| {
                    let (evals, _, pars) = self.calculate_at_k_point(k_point, n_rel_states, n_bands, select_parity);
                    progress_bar.inc(1); // Increment the progress bar
                    (evals, pars)
                })
                .unzip()
        });

        // Mark the progress bar as finished
        progress_bar.finish();

        // Calculate elapsed time
        let elapsed = start_time.elapsed();
        
        // If you want more detailed timing:
        let seconds = elapsed.as_secs();
        let millis = elapsed.subsec_millis();
        println!("Time taken: {} seconds and {} milliseconds using {} threads", 
                seconds, millis, num_threads);

        
        // Reconstruct full band structure using symmetry
        let mut eigvals_list = Vec::new();
        let mut parities_list = Vec::new();
        
        for i in 0..k_points.len() {
            let (unique_idx, _) = k_point_to_unique[&i];
            // For energy eigenvalues, symmetry operations don't change the values
            eigvals_list.push(unique_eigvals[unique_idx].clone());
            parities_list.push(unique_parities[unique_idx].clone());
        }
        
        // Save the results to file
        if self.verbose {
            info!("Saving band structure data to {}", filename);
        }
        
        // In a real implementation, we'd serialize to JSON
        
        (k_points, eigvals_list, parities_list)
    }
}
