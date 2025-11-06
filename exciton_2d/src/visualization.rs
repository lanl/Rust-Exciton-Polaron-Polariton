//! Visualization module for plotting exciton properties
//!
//! This module provides functions for visualizing eigenstates, spectra,
//! and band structures of excitons in 2D materials.

use crate::grid::Grid;
use log::info;
use ndarray::{Array1, Array2};
use plotters::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

/// Plots a relative coordinate eigenstate as a 2D heatmap
///
/// This function visualizes an eigenstate of the relative coordinate Hamiltonian
/// as a 2D heatmap, showing the probability density of the electron-hole pair.
///
/// # Arguments
/// * `grid` - The spatial grid for discretization
/// * `eigvec` - Eigenstate vector to plot
/// * `title` - Optional title for the plot
/// * `save_path` - Optional path to save the plot
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

/// Plots the energy spectrum of the exciton
///
/// This function creates a plot of the energy eigenvalues of the exciton
/// at a specific wave vector k.
///
/// # Arguments
/// * `eigvals` - Array of eigenvalues to plot
/// * `k` - Wave vector at which the eigenvalues were calculated
/// * `save_path` - Optional path to save the plot
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

/// Plots the band structure along high-symmetry paths
///
/// This function creates a plot of the exciton band structure along
/// high-symmetry paths in the Brillouin zone, with bands colored by parity.
///
/// # Arguments
/// * `k_points` - List of k-points along the path
/// * `eigvals_list` - List of eigenvalues at each k-point
/// * `parities_list` - List of parity values at each k-point
/// * `a0` - Lattice constant
/// * `save_path` - Optional path to save the plot
/// * `_show` - Whether to display the plot (currently unused)
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

/// Plots band structure with color-coded n-expectation values using gnuplot
///
/// This function creates a high-quality plot of the exciton band structure
/// with bands colored according to their n-expectation values, which indicate
/// the character of each band. It uses gnuplot for rendering.
///
/// # Arguments
/// * `k_points` - List of k-points along the path
/// * `eigvals_list` - List of eigenvalues at each k-point
/// * `n_expectation_list` - List of n-expectation values at each k-point
/// * `n_bands` - Number of bands to plot
/// * `a0` - Lattice constant
/// * `save_path` - Path to save the plot
/// * `v0` - Potential strength
///
/// # Returns
/// Result indicating success or an error
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