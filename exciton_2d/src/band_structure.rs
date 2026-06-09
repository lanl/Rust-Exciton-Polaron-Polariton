//! Band structure calculation module
//!
//! This module provides functionality for calculating band structures and
//! related properties of excitons in 2D materials.

use crate::hamiltonian::FullHamiltonian;
use crate::utils::apply_symmetry;
use log::info;
use matrix_ops::{MatrixOps, Diagonalize, ConvertFrom, MatrixOpsError};
use num_complex::Complex64;
use ndarray::{Array1, Array2};
use core::panic;
use std::collections::HashMap;
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use serde::{Serialize, Deserialize};
use bincode::{serialize, deserialize};
use std::error::Error;

/// Data structure for saving/loading band structure data with n-expectation values
///
/// This struct provides serialization and deserialization functionality for band structure
/// data, including eigenvalues and n-expectation values at each k-point. It handles
/// conversion between the serializable format and the format used in calculations.
#[derive(Serialize, Deserialize)]
pub struct ColorBandData {
    pub k_points: Vec<Vec<f64>>,
    pub eigenvalues: Vec<Vec<f64>>,  // Flattened Array1 to make serialization easier
    pub n_expectation: Vec<Vec<f64>>,
}

impl ColorBandData {
    /// Creates a new ColorBandData instance from calculation results
    ///
    /// # Arguments
    /// * `k_points` - List of k-points along the path in the Brillouin zone
    /// * `eigenvalues` - List of eigenvalues at each k-point
    /// * `n_expectation` - List of n-expectation values at each k-point
    ///
    /// # Returns
    /// A new ColorBandData instance containing the provided data
    /// Convert from calculation results to serializable format
    pub fn from_results(
        k_points: Vec<Vec<f64>>,
        eigenvalues: Vec<Array1<f64>>,
        n_expectation: Vec<Vec<f64>>
    ) -> Self {
        // Convert Array1 to Vec for serialization
        let eigenvalues_vec: Vec<Vec<f64>> = eigenvalues.iter()
            .map(|arr| arr.to_vec())
            .collect();
            
        Self {
            k_points,
            eigenvalues: eigenvalues_vec,
            n_expectation,
        }
    }
    
    /// Convert back to calculation result format
    /// Converts the stored data back to the format used in calculations
    ///
    /// # Returns
    /// A tuple containing:
    /// * List of k-points
    /// * List of eigenvalues at each k-point
    /// * List of n-expectation values at each k-point
    pub fn to_results(&self) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>) {
        // Convert Vec back to Array1
        let eigenvalues: Vec<Array1<f64>> = self.eigenvalues.iter()
            .map(|vec| Array1::from_vec(vec.clone()))
            .collect();
            
        (self.k_points.clone(), eigenvalues, self.n_expectation.clone())
    }
    
    /// Save data to file
    /// Saves the band structure data to a binary file
    ///
    /// # Arguments
    /// * `filename` - Path to the file where data should be saved
    ///
    /// # Returns
    /// Result indicating success or an error
    pub fn save_to_file(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        let data = serialize(&self)?;
        std::fs::write(filename, data)?;
        Ok(())
    }
    
    /// Load data from file
    /// Loads band structure data from a binary file
    ///
    /// # Arguments
    /// * `filename` - Path to the file containing saved data
    ///
    /// # Returns
    /// Result containing the loaded data or an error
    pub fn load_from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let data = std::fs::read(filename)?;
        let band_data: Self = deserialize(&data)?;
        Ok(band_data)
    }
}

/// Calculates the expectation value of the principal quantum number (n) for an eigenstate
///
/// This function computes the expectation value of the principal quantum number n
/// for a given eigenstate of the full Hamiltonian. This value indicates which
/// relative coordinate state contributes most to the eigenstate.
///
/// Note: Eigenvectors are complex due to -i*sin matrix elements.
pub fn calculate_n_expectation<M: MatrixOps>(
    full_ham: &FullHamiltonian<M>,
    eigenvector: &Array1<num_complex::Complex64>,
    n_rel_states: usize
) -> f64 {
    // Get the number of center-of-mass states
    let n_com_states = full_ham.kappa_vectors.len();
    
    // Get the mapping from relative state index to n quantum number
    // Generate all states in order
    let mut n_values: Vec<i32> = Vec::new();
    let mut n: i32 = 0;
    while n_values.len() < n_rel_states {
        for _m in -n..=n {
            n_values.push(n);
            if n_values.len() >= n_rel_states {
                break;
            }
        }
        n += 1;
    }
    
    // Calculate the expectation value of n
    let mut n_expectation = 0.0;
    let mut total_prob = 0.0;
    
    for i_rel in 0..n_rel_states {
        for i_com in 0..n_com_states {
            let i = i_rel * n_com_states + i_com;
            if i < eigenvector.len() {
                // For complex eigenvectors, probability is |ψ|²
                let prob = eigenvector[i].norm_sqr();
                n_expectation += prob * n_values[i_rel] as f64;
                total_prob += prob;
            }
        }
    }
    
    // Return the expectation value
    if total_prob > 0.0 {
        n_expectation / total_prob
    } else {
        0.0
    }
}

/// Calculates band structure with n-expectation values for color coding
///
/// This function computes the band structure along high-symmetry paths in the
/// Brillouin zone, including the expectation value of the principal quantum number
/// for each eigenstate. These expectation values can be used to color-code the
/// bands to visualize their character.
///
/// Note: With sine matrix elements, parity is not conserved. All states are included.
pub fn calculate_n_colored_band_structure<M: MatrixOps>(
    full_ham: &FullHamiltonian<M>,
    n_points: usize,
    n_rel_states: usize,
    n_bands: usize,
    a0: f64,
    v0: f64,
    save_dir: &str,
    recalculate: bool
) -> Result<(Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>), MatrixOpsError> 
where 
    M::Scalar: ConvertFrom<Complex64>,
    M: Diagonalize,
{
    // Create save directory if it doesn't exist
    std::fs::create_dir_all(save_dir).unwrap_or_else(|_| {});
    
    // Generate filename for the color band structure data
    let filename = format!(
        "{}/n_colored_band_a0{}_V0{}_nrel{}_nbands{}_npoints{}.bin",
        save_dir,
        full_ham.rel_ham.grid.box_size,
        v0,
        n_rel_states,
        n_bands,
        n_points
    );
    
    // Check if cached data exists and should be used
    if !recalculate && Path::new(&filename).exists() {
        info!("Loading cached n-colored band structure data from {}", filename);
        match ColorBandData::load_from_file(&filename) {
            Ok(data) => return Ok(data.to_results()),
            Err(e) => info!("Failed to load cached data: {}, recalculating...", e)
        }
    }

    // Define high-symmetry points in the Brillouin zone
    // For a 2D square lattice:
    let b1 = 2.0 * std::f64::consts::PI / a0;  // Reciprocal lattice constant
    
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
    info!("Identifying unique K points using symmetry...");
    
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
    
    info!("Reduced from {} to {} unique K points using symmetry",
          k_points.len(), unique_k_points.len());
    
    // Calculate eigenvalues for unique K points
    info!("Calculating band structure at {} unique K points...", unique_k_points.len());

    // Create a progress bar
    let progress_bar = ProgressBar::new(unique_k_points.len() as u64);
    progress_bar.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} k-points ({eta})")
        .unwrap()
        .progress_chars("#>-"));
    
    // Store original K value
    let original_k = full_ham.k.clone();
    
    // Calculate eigenvalues and n-expectation values for each unique k-point
    let mut unique_results = Vec::new();
    
    for (idx, k_point) in unique_k_points.iter().enumerate() {
        // Create a thread-local copy of the FullHamiltonian
        let mut full_ham_local = full_ham.clone();
        full_ham_local.k = k_point.clone();
        
        // Diagonalize the Hamiltonian
        let (eigvals, eigvecs, _,_) = 
            full_ham_local.diagonalize_full_hamiltonian(n_rel_states, n_bands)?;
        
        // Calculate n-expectation for each eigenstate
        let n_expectation: Vec<f64> = (0..eigvals.len())
            .map(|i| calculate_n_expectation(
                &full_ham_local, 
                &eigvecs.column(i).to_owned(),
                n_rel_states
            ))
            .collect();
        
        // Store results
        unique_results.push((idx, eigvals, n_expectation));
        
        // Update progress bar
        progress_bar.inc(1);
    }
    
    progress_bar.finish_with_message("Band structure calculation complete");
    
    // Restore original K value
    let mut full_ham_local = full_ham.clone();
    full_ham_local.k = original_k;
    
    // Reconstruct full band structure using symmetry
    let mut eigvals_list = vec![Array1::<f64>::zeros(0); k_points.len()];
    let mut n_expectation_list = vec![Vec::<f64>::new(); k_points.len()];
    
    for i in 0..k_points.len() {
        let (unique_idx, _) = k_point_to_unique[&i];
        
        // Find the corresponding unique result
        for (idx, eigvals, n_expectation) in &unique_results {
            if *idx == unique_idx {
                eigvals_list[i] = eigvals.iter().map(|&val| val.re).collect();
                n_expectation_list[i] = n_expectation.clone();
                break;
            }
        }
    }
    
    // Save the results to file
    let band_data = ColorBandData::from_results(k_points.clone(), eigvals_list.clone(), n_expectation_list.clone());
    match band_data.save_to_file(&filename) {
        Ok(_) => info!("N-colored band structure data saved to {}", filename),
        Err(e) => info!("Failed to save n-colored band structure data: {}", e)
    }

    Ok((k_points, eigvals_list, n_expectation_list))
}

/// Main calculator for band structure calculations
///
/// This struct provides methods for calculating band structures of excitons
/// in 2D materials with periodic potentials. It handles the calculation of
/// eigenvalues and eigenvectors at different k-points in the Brillouin zone.
pub struct BandStructureCalculator<M: MatrixOps> {
    pub full_ham: FullHamiltonian<M>,
    pub a0: f64,
    pub save_dir: String,
    pub verbose: bool,
}

impl<M: MatrixOps> BandStructureCalculator<M> 
where 
    M::Scalar: ConvertFrom<Complex64>,
    M: Diagonalize,
{
    /// Creates a new band structure calculator
    ///
    /// # Arguments
    /// * `full_ham` - Full Hamiltonian for the exciton
    /// * `a0` - Lattice constant
    /// * `save_dir` - Directory to save/load band structure data
    /// * `verbose` - Whether to print detailed information during calculations
    ///
    /// # Returns
    /// A new BandStructureCalculator instance
    pub fn new(full_ham: FullHamiltonian<M>, a0: f64, save_dir: String, verbose: bool) -> Self {
        Self {
            full_ham,
            a0,
            save_dir,
            verbose,
        }
    }
    
    /// Generates a filename for saving/loading band structure data
    ///
    /// # Arguments
    /// * `n_points` - Number of k-points along each path segment
    /// * `n_rel_states` - Number of relative coordinate states included
    /// * `n_bands` - Number of bands calculated
    /// * `v0` - Potential strength
    ///
    /// # Returns
    /// A filename string based on the calculation parameters
    pub fn get_band_structure_filename(
        &self,
        n_points: usize,
        n_rel_states: usize,
        n_bands: usize,
        v0: f64,
    ) -> String {
        format!(
            "{}/band_a0{}_V0{}_grid{}_box{}_nrel{}_nbands{}_npoints{}.json",
            self.save_dir,
            self.a0,
            v0,
            self.full_ham.rel_ham.grid.grid_size,
            self.full_ham.rel_ham.grid.box_size,
            n_rel_states,
            n_bands,
            n_points
        )
    }
    
    /// Calculates eigenvalues and eigenvectors at a specific k-point
    ///
    /// Note: With sine matrix elements, parity is not conserved. All states are included.
    ///
    /// # Arguments
    /// * `k_point` - Wave vector in the Brillouin zone
    /// * `n_rel_states` - Number of relative coordinate states to include
    /// * `n_bands` - Number of bands to calculate
    ///
    /// # Returns
    /// A tuple containing eigenvalues, eigenvectors, and parity values (of unperturbed states)
    pub fn calculate_at_k_point(
        &self,
        k_point: &[f64],
        n_rel_states: usize,
        n_bands: usize,
    ) -> Result<(Array1<Complex64>, Array2<Complex64>), MatrixOpsError> {
        // Store original K value
        let original_k = self.full_ham.k.clone();
        
        // Set the new K value
        let mut full_ham = self.full_ham.clone();
        full_ham.k = k_point.to_vec();
        
        // Diagonalize the Hamiltonian
        let (eigvals, eigvecs, _, _) = full_ham.diagonalize_full_hamiltonian(
            n_rel_states,
            n_bands,
        )?;
        
        // Restore original K value
        let mut full_ham = self.full_ham.clone();
        full_ham.k = original_k;
        
        Ok((eigvals, eigvecs))
    }
    
    /// Calculates the band structure along high-symmetry paths
    ///
    /// This method computes the band structure along the path Γ→X→M→Γ in the
    /// Brillouin zone. It uses symmetry to reduce the number of calculations
    /// and can parallelize the computation across multiple threads.
    ///
    /// Note: With sine matrix elements, parity is not conserved. All states are included.
    ///
    /// # Arguments
    /// * `n_points` - Number of k-points along each path segment
    /// * `n_rel_states` - Number of relative coordinate states to include
    /// * `n_bands` - Number of bands to calculate
    /// * `v0` - Potential strength
    /// * `recalculate_bands` - Whether to force recalculation
    ///
    /// # Returns
    /// A tuple containing k-points, eigenvalues, and parity values (of unperturbed states)
    pub fn calculate_band_structure(
        &self,
        n_points: usize,
        n_rel_states: usize,
        n_bands: usize,
        v0: f64,
        recalculate_bands: bool,
    ) -> (Vec<Vec<f64>>, Vec<Array1<f64>>) {
        // Check if we should try to load from file
        let filename = self.get_band_structure_filename(n_points, n_rel_states, n_bands, v0);
        
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

        let mut unique_eigvals = Vec::with_capacity(unique_k_points.len());

        for k_point in &unique_k_points {
            let (evals, _) = self
                .calculate_at_k_point(k_point, n_rel_states, n_bands)
                .unwrap();

            unique_eigvals.push(
                evals.iter()
                    .map(|&val| val.re)
                    .collect::<Array1<f64>>()
            );

            progress_bar.inc(1);
        }

        // Mark the progress bar as finished
        progress_bar.finish();

        // Calculate elapsed time
        let elapsed = start_time.elapsed();
        
        // If you want more detailed timing:
        let seconds = elapsed.as_secs();
        let millis = elapsed.subsec_millis();
        println!("Time taken: {} seconds and {} milliseconds", 
                seconds, millis);

        
        // Reconstruct full band structure using symmetry
        let mut eigvals_list = Vec::new();
        
        for i in 0..k_points.len() {
            let (unique_idx, _) = k_point_to_unique[&i];
            // For energy eigenvalues, symmetry operations don't change the values
            eigvals_list.push(unique_eigvals[unique_idx].clone());
        }
        
        // Save the results to file
        if self.verbose {
            info!("Saving band structure data to {}", filename);
        }
        
        // In a real implementation, we'd serialize to JSON
        
        (k_points, eigvals_list)
    }
}
