// src/visualization.rs
use crate::grid::Grid;
use crate::hamiltonian::FullHamiltonian;
use crate::utils::apply_symmetry;
use log::info;
use ndarray::{Array1, Array2};
use plotters::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use serde::{Serialize, Deserialize};
use bincode::{serialize, deserialize};
use std::error::Error;
use indicatif::{ProgressBar, ProgressStyle};

pub fn plot_rel_eigenstate(grid: &Grid, eigvec: &Array1<f64>, title: Option<&str>, save_path: Option<&str>) {
    // This is a placeholder for the visualization functionality
    // In a real implementation, we would use plotters to create the visualization
    
    if let Some(path) = save_path {
        info!("Saving eigenstate plot to {}", path);
    }
    
    // Example of how we would implement this with plotters:
    if let Some(path) = save_path {
        let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        
        // Create a 2D heatmap of the eigenstate
        let psi = eigvec.to_shape((grid.grid_size, grid.grid_size)).unwrap().to_owned();
        
        // Calculate probability density
        let mut prob_density = Array2::zeros((grid.grid_size, grid.grid_size));
        for i in 0..grid.grid_size {
            for j in 0..grid.grid_size {
                prob_density[[i, j]] = psi[[i, j]].powi(2);
            }
        }
        
        // Set up chart
        let mut chart = ChartBuilder::on(&root)
            .caption(title.unwrap_or("Relative Coordinate Eigenstate"), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                grid.x[0]..grid.x[grid.grid_size - 1],
                grid.y[0]..grid.y[grid.grid_size - 1]
            )
            .unwrap();
        
        // Draw heatmap
        chart.configure_mesh().draw().unwrap();
        
        // In a real implementation, we would draw the heatmap here
        
        root.present().unwrap();
    }
}

pub fn plot_spectrum(eigvals: &Array1<f64>, k: &[f64], save_path: Option<&str>) {
    // This is a placeholder for the spectrum plotting functionality
    
    if let Some(path) = save_path {
        info!("Saving spectrum plot to {}", path);
        
        let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        
        let mut chart = ChartBuilder::on(&root)
            .caption(format!("Exciton energy spectrum at K = {:?}", k), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0.0..eigvals.len() as f64,
                eigvals.fold(f64::INFINITY, |a, &b| a.min(b))..eigvals.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            )
            .unwrap();
        
        chart.configure_mesh()
            .x_desc("State index")
            .y_desc("Energy")
            .draw()
            .unwrap();
        
        // Draw points
        chart.draw_series(
            (0..eigvals.len()).map(|i| Circle::new((i as f64, eigvals[i]), 3, BLUE.filled()))
        ).unwrap();
        
        root.present().unwrap();
    }
}

pub fn plot_band_structure(
    k_points: &[Vec<f64>],
    eigvals_list: &[Array1<f64>],
    parities_list: &[Array1<i32>],
    a0: f64,
    save_path: Option<&str>,
    _show: bool,
) {
    // This is a placeholder for the band structure plotting functionality
    
    if let Some(path) = save_path {
        info!("Saving band structure plot to {}", path);
        
        let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        
        // Create x-axis for the plot (path length)
        let mut x = vec![0.0; k_points.len()];
        for i in 1..k_points.len() {
            let dx = k_points[i][0] - k_points[i-1][0];
            let dy = k_points[i][1] - k_points[i-1][1];
            x[i] = x[i-1] + (dx*dx + dy*dy).sqrt();
        }
        
        // Define the high-symmetry points
        let _b1 = 2.0 * std::f64::consts::PI / a0;
        let points_per_segment = k_points.len() / 3;
        
        // Positions of high-symmetry points
        let x_gamma1 = x[0];
        let x_k = x[points_per_segment - 1];
        let x_m = x[2 * points_per_segment - 1];
        let x_gamma2 = x[x.len() - 1];
        
        // Find min/max energy values
        let mut min_energy = f64::INFINITY;
        let mut max_energy = f64::NEG_INFINITY;
        
        for eigvals in eigvals_list {
            min_energy = min_energy.min(eigvals.fold(f64::INFINITY, |a, &b| a.min(b)));
            max_energy = max_energy.max(eigvals.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        }
        
        // Set up chart
        let mut chart = ChartBuilder::on(&root)
            .caption("Exciton Band Structure", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                x[0]..x[x.len() - 1],
                min_energy..max_energy
            )
            .unwrap();
        
        // Configure mesh with custom x-ticks for high-symmetry points
        chart.configure_mesh()
            .x_labels(4)
            .x_label_formatter(&|v| {
                if (v - &x_gamma1).abs() < 1e-6 { "Γ".to_string() }
                else if (v - &x_k).abs() < 1e-6 { "X".to_string() }
                else if (v - &x_m).abs() < 1e-6 { "M".to_string() }
                else if (v - &x_gamma2).abs() < 1e-6 { "Γ".to_string() }
                else { "".to_string() }
            })
            .y_desc("Energy")
            .draw()
            .unwrap();
        
        // Draw vertical lines at high-symmetry points
        chart.draw_series(vec![
            PathElement::new(vec![(x_gamma1, min_energy), (x_gamma1, max_energy)], BLACK.mix(0.5)),
            PathElement::new(vec![(x_k, min_energy), (x_k, max_energy)], BLACK.mix(0.5)),
            PathElement::new(vec![(x_m, min_energy), (x_m, max_energy)], BLACK.mix(0.5)),
            PathElement::new(vec![(x_gamma2, min_energy), (x_gamma2, max_energy)], BLACK.mix(0.5)),
        ]).unwrap();
        
        // Draw each band with colors based on parity values
        for band_idx in 0..eigvals_list[0].len() {
            for k_idx in 0..x.len() - 1 {
                // Get parity value for this segment
                let parity_val = (parities_list[k_idx][band_idx] + parities_list[k_idx+1][band_idx]) / 2;
                let color = if parity_val > 0 { BLUE } else { RED };
                
                // Plot the segment with the appropriate color
                chart.draw_series(LineSeries::new(
                    vec![
                        (x[k_idx], eigvals_list[k_idx][band_idx]),
                        (x[k_idx+1], eigvals_list[k_idx+1][band_idx])
                    ],
                    color.mix(0.8)
                )).unwrap();
            }
        }
        
        root.present().unwrap();
    }
}



/// Calculate the expectation value of the principal quantum number (n) for an eigenstate
pub fn calculate_n_expectation(
    full_ham: &FullHamiltonian,
    eigenvector: &Array1<f64>,
    parity: i32,
    n_rel_states: usize
) -> f64 {
    // Get the number of center-of-mass states
    let n_com_states = full_ham.kappa_vectors.len();
    
    // Get the mapping from relative state index to (n,m) quantum numbers
    // This is a simplified approach - in a real implementation, you would need to 
    // extract this information from your relative Hamiltonian solution
    let mut n_values_both: (Vec<i32>,Vec<i32>) = (Vec::new(),Vec::new());
    let mut n: i32 = 0;
    while (n_values_both.0.len() + n_values_both.1.len()) < n_rel_states {
        for m in -n..=n {
            if n >= m.abs() {  // Valid state condition
                if m % 2 == 0 {
                    n_values_both.0.push(n);
                }
                else {
                    n_values_both.1.push(n);
                }
            }
        }
        n += 1;
    }
    
    let n_values: Vec<i32>;
    
    if parity == 1 {
        n_values = n_values_both.0;
    }
    else {
        n_values = n_values_both.1;
    }
    
    // Calculate the expectation value of n
    let mut n_expectation = 0.0;
    let mut total_prob = 0.0;
    
    for i_rel in 0..n_rel_states {
        for i_com in 0..n_com_states {
            let i = i_rel * n_com_states + i_com;
            if i < eigenvector.len() {
                let prob = eigenvector[i] * eigenvector[i];
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

/// Data structure for saving/loading color band structure data
#[derive(Serialize, Deserialize)]
pub struct ColorBandData {
    pub k_points: Vec<Vec<f64>>,
    pub eigenvalues: Vec<Vec<f64>>,  // Flattened Array1 to make serialization easier
    pub n_expectation: Vec<Vec<f64>>,
}

impl ColorBandData {
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
    pub fn to_results(&self) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>) {
        // Convert Vec back to Array1
        let eigenvalues: Vec<Array1<f64>> = self.eigenvalues.iter()
            .map(|vec| Array1::from_vec(vec.clone()))
            .collect();
            
        (self.k_points.clone(), eigenvalues, self.n_expectation.clone())
    }
    
    /// Save data to file
    pub fn save_to_file(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        let data = serialize(&self)?;
        std::fs::write(filename, data)?;
        Ok(())
    }
    
    /// Load data from file
    pub fn load_from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let data = std::fs::read(filename)?;
        let band_data: Self = deserialize(&data)?;
        Ok(band_data)
    }
}

/// Calculate band structure with n-expectation values
pub fn calculate_n_colored_band_structure(
    full_ham: &FullHamiltonian,
    n_points: usize,
    n_rel_states: usize,
    n_bands: usize,
    a0: f64,
    v0: f64,
    select_parity: i32,
    save_dir: &str,
    recalculate: bool
) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>) {
    // Create save directory if it doesn't exist
    std::fs::create_dir_all(save_dir).unwrap_or_else(|_| {});
    
    // Generate filename for the color band structure data
    let filename = format!(
        "{}/n_colored_band_a0{}_V0{}_nrel{}_nbands{}_npoints{}_par{}.bin",
        save_dir,
        full_ham.rel_ham.grid.box_size,
        v0,
        n_rel_states,
        n_bands,
        n_points,
        select_parity
    );
    
    // Check if cached data exists and should be used
    if !recalculate && Path::new(&filename).exists() {
        info!("Loading cached n-colored band structure data from {}", filename);
        match ColorBandData::load_from_file(&filename) {
            Ok(data) => return data.to_results(),
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
        let (eigvals, eigvecs, _,_, parities) = 
            full_ham_local.diagonalize_full_hamiltonian(n_rel_states, n_bands, select_parity);
        
        // Calculate n-expectation for each eigenstate
        let n_expectation: Vec<f64> = (0..eigvals.len())
            .map(|i| calculate_n_expectation(
                &full_ham_local, 
                &eigvecs[i], 
                parities[i],
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
                eigvals_list[i] = eigvals.clone();
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

    (k_points, eigvals_list, n_expectation_list)
}

/// Plot band structure with color-coded n-expectation values using gnuplot
pub fn plot_n_colored_band_structure(
    k_points: &Vec<Vec<f64>>,
    eigvals_list: &Vec<Array1<f64>>,
    n_expectation_list: &Vec<Vec<f64>>,
    n_bands: usize,
    a0: f64,
    save_path: &str,
    v0: f64
) -> Result<(), Box<dyn std::error::Error>> {
    // Create x-axis for the plot (path length)
    let mut x = vec![0.0; k_points.len()];
    for i in 1..k_points.len() {
        let dx = k_points[i][0] - k_points[i-1][0];
        let dy = k_points[i][1] - k_points[i-1][1];
        x[i] = x[i-1] + (dx*dx + dy*dy).sqrt();
    }

    // Define the high-symmetry points
    let points_per_segment = k_points.len() / 3;
    
    // Positions of high-symmetry points
    let x_gamma1 = x[0];
    let x_k = x[points_per_segment - 1];
    let x_m = x[2 * points_per_segment - 1];
    let x_gamma2 = x[x.len() - 1];
    
    // Find min/max energy values
    let mut min_energy = f64::INFINITY;
    let mut max_energy = f64::NEG_INFINITY;
    
    // Find min/max n-expectation values
    let mut min_n = f64::INFINITY;
    let mut max_n = f64::NEG_INFINITY;
    
    // Limit the number of bands to plot
    let n_bands = n_bands.min(eigvals_list[0].len());

    // Find energy and n-expectation ranges
    for (i, eigvals) in eigvals_list.iter().enumerate() {
        let slice = eigvals.slice(ndarray::s![0..n_bands]);
        min_energy = min_energy.min(slice.fold(f64::INFINITY, |a, &b| a.min(b)));
        max_energy = max_energy.max(slice.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        
        for j in 0..n_bands.min(n_expectation_list[i].len()) {
            min_n = min_n.min(n_expectation_list[i][j]);
            max_n = max_n.max(n_expectation_list[i][j]);
        }
    }
    
    // Add padding to the energy range (5%)
    let energy_padding = (max_energy - min_energy) * 0.05;

    max_energy = 0.0;

    min_energy -= energy_padding;
    max_energy += energy_padding;

    // Create data files for gnuplot
    let data_dir = Path::new("gnuplot_data");
    std::fs::create_dir_all(data_dir)?;
    
    // Create a data file for each band
    for band_idx in 0..n_bands {
        let filename = data_dir.join(format!("n_band_{}.dat", band_idx));
        let mut file = File::create(filename)?;
        
        // Write header
        writeln!(file, "# x energy n_expectation")?;
        
        // Write data points
        for (i, &x_val) in x.iter().enumerate() {
            if band_idx < eigvals_list[i].len() && band_idx < n_expectation_list[i].len() {
                let energy = eigvals_list[i][band_idx];
                let n_exp = n_expectation_list[i][band_idx];
                
                writeln!(file, "{} {} {}", x_val, energy, n_exp)?;
            }
        }
    }
    
    // Create gnuplot script
    let script_path = data_dir.join("plot_n_bands.gp");
    let mut script = File::create(&script_path)?;
    
    // Write gnuplot script
    writeln!(script, "set terminal pngcairo enhanced font 'Arial,40' size 2400,1800")?;
    writeln!(script, "set output '{}'", save_path)?;
    
    // Set margins to make room for the colorbar
    writeln!(script, "set lmargin 10")?;
    writeln!(script, "set rmargin 10")?;
    writeln!(script, "set tmargin 3")?;
    writeln!(script, "set bmargin 3")?;
    
    writeln!(script, "set title 'Exciton Band Structure (a₀={:.1}, V₀={:.1})' font 'Arial,16'", a0, v0)?;
    writeln!(script, "set xlabel 'Wave vector'")?;
    writeln!(script, "set ylabel 'Energy (a.u.)'")?;
    writeln!(script, "set xrange [{}:{}]", x[0], x[x.len() - 1])?;
    writeln!(script, "set yrange [{}:{}]", min_energy, max_energy)?;
    
    // Set up color palette - blue (low n) to red (high n)
    writeln!(script, "set palette rgbformulae 22,13,-22")?;
    writeln!(script, "set colorbox vertical user origin 0.88,0.1 size 0.03,0.8")?;
    writeln!(script, "set cbrange [{}:{}]", min_n, max_n)?;
    writeln!(script, "set cblabel 'Principal Quantum Number (n)' offset 2,0")?;
    writeln!(script, "set cbtics 1")?;
    
    // Set up x-axis with symmetry point labels
    writeln!(script, "set xtics ('Γ' {}, 'X' {}, 'M' {}, 'Γ' {})", 
             x_gamma1, x_k, x_m, x_gamma2)?;
    
    // Draw vertical lines at symmetry points
    for &pos in &[x_gamma1, x_k, x_m, x_gamma2] {
        writeln!(script, "set arrow from {},{} to {},{} nohead lc rgb '#888888' lw 6", 
                 pos, min_energy, pos, max_energy)?;
    }
    
    // Plot all bands with color mapping
    write!(script, "plot ")?;
    
    for band_idx in 0..n_bands {
        if band_idx > 0 {
            write!(script, ", ")?;
        }
        write!(script, "'{}' using 1:2:3 with lines lw 8 lc palette notitle", 
               data_dir.join(format!("n_band_{}.dat", band_idx)).display())?;
    }
    writeln!(script)?;
    
    // Execute gnuplot
    let status = Command::new("gnuplot")
        .arg(script_path)
        .status()?;
    
    if !status.success() {
        return Err("Failed to execute gnuplot".into());
    }
    
    info!("N-colored band structure plot saved to {}", save_path);
    Ok(())
}