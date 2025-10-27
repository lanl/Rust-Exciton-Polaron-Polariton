use log::info;
use plotters::prelude::*;
use ndarray::{Array1, s};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::collections::HashMap;

/// Plots the dielectric function for different coupling strengths
pub fn plot_dielectric_function(
    dir_path: &str,                // Directory containing dielectric function data files
    save_path: Option<&str>,       // Path to save the plot
    plot_type: DielectricPlotType, // Which part of the dielectric function to plot
    omega_range: Option<(f64, f64)>, // Optional frequency range to plot
) -> Result<(), Box<dyn std::error::Error>> {
    // Get all dielectric function files in the directory
    let dir = Path::new(dir_path);
    let mut files = std::fs::read_dir(dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_file() && path.extension()?.to_str()? == "dat" {
                Some(path)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    
    // Sort files by coupling strength
    files.sort_by(|a, b| {
        let coupling_a = extract_coupling_from_filename(a).unwrap_or(0.0);
        let coupling_b = extract_coupling_from_filename(b).unwrap_or(0.0);
        coupling_a.partial_cmp(&coupling_b).unwrap()
    });
    
    if files.is_empty() {
        return Err("No dielectric function data files found".into());
    }
    
    // Load data from files
    let mut all_data: HashMap<PathBuf, (Vec<f64>, Vec<f64>, Vec<f64>)> = HashMap::new();
    let mut omega_min = f64::INFINITY;
    let mut omega_max = f64::NEG_INFINITY;
    let mut eps_min = f64::INFINITY;
    let mut eps_max = f64::NEG_INFINITY;
    
    for file_path in &files {
        let coupling = extract_coupling_from_filename(file_path);
        let (omegas, re_eps, im_eps) = load_dielectric_data(file_path)?;
        
        // Update min/max values
        if let Some(&min_omega) = omegas.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            omega_min = omega_min.min(min_omega);
        }
        if let Some(&max_omega) = omegas.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            omega_max = omega_max.max(max_omega);
        }
        
        match plot_type {
            DielectricPlotType::Real => {
                if let Some(&min_eps) = re_eps.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                    eps_min = eps_min.min(min_eps);
                }
                if let Some(&max_eps) = re_eps.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                    eps_max = eps_max.max(max_eps);
                }
            },
            DielectricPlotType::Imaginary => {
                if let Some(&min_eps) = im_eps.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                    eps_min = eps_min.min(min_eps);
                }
                if let Some(&max_eps) = im_eps.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                    eps_max = eps_max.max(max_eps);
                }
            },
            DielectricPlotType::Both => {
                if let Some(&min_re) = re_eps.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                    eps_min = eps_min.min(min_re);
                }
                if let Some(&min_im) = im_eps.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                    eps_min = eps_min.min(min_im);
                }
                if let Some(&max_re) = re_eps.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                    eps_max = eps_max.max(max_re);
                }
                if let Some(&max_im) = im_eps.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                    eps_max = eps_max.max(max_im);
                }
            },
        }

        // eps_max = eps_max.min( 100.0);
        // eps_min = eps_min.max(-100.0);
        
        println!("Coupling = {:?}", coupling);
        println!("Max eps = {}", eps_max);
        println!("Min eps = {}", eps_min);
        all_data.insert(file_path.clone(), (omegas, re_eps, im_eps));
    }
    
    // Apply omega range if specified
    if let Some((min, max)) = omega_range {
        omega_min = min;
        omega_max = max;
    }
    
    // Add padding to the eps range (5%)
    let eps_padding = (eps_max - eps_min) * 0.05;
    eps_min -= eps_padding;
    eps_max += eps_padding;
    
    // Create the plot
    let save_path = save_path.unwrap_or("dielectric_function.png");
    let root = BitMapBackend::new(save_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Split the drawing area for multiple plots if needed
    let areas = if plot_type == DielectricPlotType::Both {
        let (top, bottom) = root.split_vertically(450);
        vec![top, bottom]
    } else {
        vec![root.clone()]
    };
    
    // Get sorted couplings
    let mut couplings: Vec<(PathBuf,f64)> = all_data.keys().map(|file_path| {
        (file_path.clone(), extract_coupling_from_filename(file_path).unwrap_or(0.0))
    }).collect();
    couplings.sort_by(|(_,a), (_,b)| a.partial_cmp(b).unwrap());
    
    // Plot title
    let title = match plot_type {
        DielectricPlotType::Real => "Real Part of Dielectric Function",
        DielectricPlotType::Imaginary => "Imaginary Part of Dielectric Function",
        DielectricPlotType::Both => "Dielectric Function",
    };
    
    // Plot the real part
    if plot_type == DielectricPlotType::Real || plot_type == DielectricPlotType::Both {
        let area_index = if plot_type == DielectricPlotType::Both { 0 } else { 0 };
        let mut chart = ChartBuilder::on(&areas[area_index])
            .caption(format!("{} - Real Part", title), ("sans-serif", 30).into_font())
            .margin(10)
            .margin_top(20)
            .margin_bottom(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(omega_min..omega_max, eps_min..eps_max)?;
        
        chart.configure_mesh()
            .x_desc("Frequency (a.u.)")
            .y_desc("Re[ε(ω)]")
            .axis_desc_style(("sans-serif", 20))
            .label_style(("sans-serif", 18))
            .draw()?;
        

        
        for (i, (key, coupling)) in couplings.iter().enumerate() {
            let (omegas, re_eps, _) = &all_data[key];
            
            // Create a color gradient from blue to red
            let color = RGBColor(
                ((1.0 - i as f64 / couplings.len() as f64) * 255.0) as u8, // More red for higher coupling
                0,
                ((i as f64 / couplings.len() as f64) * 255.0) as u8  // More blue for lower coupling
            );
            
            // Filter data points within the omega range
            let points: Vec<(f64, f64)> = omegas.iter()
                .zip(re_eps.iter())
                .filter(|(omega, _)| omega >= &&omega_min && omega <= &&omega_max)
                .map(|(&omega, &eps)| (omega, eps))
                .collect();
            
            chart.draw_series(LineSeries::new(
                points,
                ShapeStyle {
                    color: color.into(),
                    filled: false,
                    stroke_width: 3,
                }
            ))?.label(format!("g = {:.3}", coupling))
              .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }
         
        // Add legend
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .label_font(("sans-serif", 15))
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }
    
    // Plot the imaginary part
    if plot_type == DielectricPlotType::Imaginary || plot_type == DielectricPlotType::Both {
        let area_index = if plot_type == DielectricPlotType::Both { 1 } else { 0 };
        let mut chart = ChartBuilder::on(&areas[area_index])
            .caption(format!("{} - Imaginary Part", title), ("sans-serif", 30).into_font())
            .margin(10)
            .margin_top(20)
            .margin_bottom(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(omega_min..omega_max, eps_min..eps_max)?;
        
        chart.configure_mesh()
            .x_desc("Frequency (a.u.)")
            .y_desc("Im[ε(ω)]")
            .axis_desc_style(("sans-serif", 20))
            .label_style(("sans-serif", 18))
            .draw()?;
        
        // Draw each coupling strength with a different color
        for (i, (key, coupling)) in couplings.iter().enumerate() {
            let (omegas, _, im_eps) = &all_data[key];
            
            // Create a color gradient from blue to red
            let color = RGBColor(
                ((1.0 - i as f64 / couplings.len() as f64) * 255.0) as u8, // More red for higher coupling
                0,
                ((i as f64 / couplings.len() as f64) * 255.0) as u8  // More blue for lower coupling
            );
            
            // Filter data points within the omega range
            let points: Vec<(f64, f64)> = omegas.iter()
                .zip(im_eps.iter())
                .filter(|(omega, _)| omega >= &&omega_min && omega <= &&omega_max)
                .map(|(&omega, &eps)| (omega, eps))
                .collect();
            
            chart.draw_series(LineSeries::new(
                points,
                ShapeStyle {
                    color: color.into(),
                    filled: false,
                    stroke_width: 3,
                }
            ))?.label(format!("g = {:.3}", coupling))
              .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }
        
        // Add legend
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .label_font(("sans-serif", 15))
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }
    
    // Present the final plot
    root.present()?;
    info!("Dielectric function plot saved to {}", save_path);
    
    Ok(())
}

/// Helper function to extract coupling strength from filename
fn extract_coupling_from_filename(path: &Path) -> Option<f64> {
    let filename = path.file_name()?.to_str()?;
    let parts: Vec<&str> = filename.split('_').collect();
    
    for (i, part) in parts.iter().enumerate() {
        if *part == "coupling" && i + 1 < parts.len() {
            return parts[i + 1].trim_end_matches(".dat").parse::<f64>().ok();
        }
    }
    
    None
}

/// Helper function to load dielectric function data from a file
fn load_dielectric_data(path: &Path) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    let mut omegas = Vec::new();
    let mut re_eps = Vec::new();
    let mut im_eps = Vec::new();
    
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        
        let values: Vec<f64> = line.split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();
        
        if values.len() >= 3 {
            omegas.push(values[0]);
            re_eps.push(values[1]);
            im_eps.push(values[2]);
        }
    }
    
    Ok((omegas, re_eps, im_eps))
}

/// Enum to specify which part of the dielectric function to plot
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DielectricPlotType {
    Real,
    Imaginary,
    Both,
}

/// Plots the band structure
pub fn plot_band_structure(
    k_points: &Vec<Vec<f64>>,           // k-points along the path
    sym_points: Vec<Vec<f64>>,          // High-symmetry points
    sym_labels: Vec<String>,            // Labels for high-symmetry points
    eigvals_list_full: &Vec<Array1<f64>>, // Eigenvalues at each k-point
    n_bands: usize,                     // Number of bands to plot
    a0: f64,                            // Lattice constant
    save_path: Option<&str>,            // Path to save the plot
    _show: bool,                        // Whether to show the plot (unused)
    v0: f64,                            // Potential strength
    coupling: f64,                      // Light-matter coupling
) {
    if let Some(path) = save_path {
        info!("Saving band structure plot to {}", path);
        
        // Create a larger drawing area with better resolution
        let root = BitMapBackend::new(path, (1200, 900)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        
        // Create x-axis for the plot (path length)
        let mut x = vec![0.0; k_points.len()];
        for i in 1..k_points.len() {
            let dx = k_points[i][0] - k_points[i-1][0];
            let dy = k_points[i][1] - k_points[i-1][1];
            x[i] = x[i-1] + (dx*dx + dy*dy).sqrt();
        }
        
        let dx = k_points[1][0] - k_points[0][0];

        // Define the high-symmetry points
        let points_per_segment = k_points.len() / 3;
        let n_sym = sym_points.len();
        
        // Calculate positions of symmetry points on the x-axis
        let mut sym_positions: Vec<f64> = sym_points.iter().enumerate().map(|(ind, _)| {
            x[ind * points_per_segment]
        }).collect();
        sym_positions.push(x[x.len() - 1]);

        // Find min/max energy values with some padding
        let mut min_energy = f64::INFINITY;
        let mut max_energy = f64::NEG_INFINITY;

        // Limit the number of bands to plot
        let n_bands = n_bands.min(eigvals_list_full[0].len());

        // Extract the bands to plot
        let eigvals_list: &Vec<Array1<f64>> = &eigvals_list_full.iter().map(|eigvals| {
            eigvals.slice(s![0..n_bands-1]).to_owned()
        }).collect();
        
        // Find energy range
        for eigvals in eigvals_list {
            min_energy = min_energy.min(eigvals.fold(f64::INFINITY, |a, &b| a.min(b)));
            max_energy = max_energy.max(eigvals.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        }
        
        // Set max energy to 0.3
        max_energy = 0.3;

        // Add padding to the energy range (5%)
        let energy_padding = (max_energy - min_energy) * 0.05;
        min_energy -= energy_padding;
        max_energy += energy_padding;
        
        // Add a title with parameters
        let title = format!("Exciton-Polariton Band Structure (a₀={:.1}, V₀={:.1}, g={:.3})", 
                           a0, v0, coupling);

        // Set up chart with better styling
        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 30).into_font().color(&BLACK))
            .margin(10)
            .margin_top(20)
            .margin_bottom(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(
                x[0]..x[x.len() - 1],
                min_energy..max_energy
            )
            .unwrap();
        
        // Configure mesh with custom x-ticks for high-symmetry points
        chart.configure_mesh()
            .x_labels(n_sym + 1)
            .x_label_formatter(&|v| {
                for (ind, &x_pos) in sym_positions.iter().enumerate() {
                    if (v - x_pos).abs() < dx {
                        if ind < n_sym {
                            return sym_labels[ind].clone();
                        } else {
                            return sym_labels[0].clone(); // Last point wraps to first label
                        }
                    }
                }
                "".to_string()
            })
            .y_desc("Energy (a.u.)")
            .axis_desc_style(("sans-serif", 20))
            .label_style(("sans-serif", 18))
            .disable_mesh()
            .draw()
            .unwrap();
        
        // Draw vertical lines at high-symmetry points
        let line_style = ShapeStyle {
            color: BLACK.mix(0.6),
            filled: false,
            stroke_width: 3,
        };
        
        for &x_pos in &sym_positions {
            chart.draw_series(LineSeries::new(
                vec![(x_pos, min_energy), (x_pos, max_energy)],
                line_style,
            )).unwrap();
        }
        
        // Draw each band with the same color (blue)
        let num_bands = (&eigvals_list[0]).len();
        println!("Number of bands: {}", num_bands );
        
        for band_idx in 0..num_bands {
            let color = BLACK;
            let band_points: Vec<(f64, f64)> = x.iter()
                .zip(eigvals_list.iter())
                .map(|(&x_val, eigvals)| (x_val, eigvals[band_idx]))
                .collect();
            
            chart.draw_series(LineSeries::new(
                band_points,
                ShapeStyle {
                    color: color.mix(1.0),
                    filled: false,
                    stroke_width: 3,
                }
            )).unwrap();
        }
        
        // Present the final plot
        root.present().unwrap();
        info!("Band structure plot saved to {}", path);
    }
}