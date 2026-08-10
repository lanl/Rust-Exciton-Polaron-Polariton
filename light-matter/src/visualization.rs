use log::info;
use ndarray::{Array1, s};
use plotters::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::{Vector3D, timed_println};

/// Structure to hold parameters extracted from dielectric function filenames
#[derive(Debug, Clone)]
pub struct DielectricParams {
    pub a0: f64,
    pub v0: f64,
    pub n_rel: usize,
    pub lm_coupling: f64,
    pub n_modes: usize,
    pub pn_coupling: f64,
    pub pn_n_fock: usize,
    pub n_eig: usize,
    pub gamma: f64,
}

/// Helper function to extract all parameters from a dielectric function filename
pub fn extract_params_from_filename(path: &Path) -> Option<DielectricParams> {
    let filename = path.file_name()?.to_str()?;

    // Skip if not a dielectric function file
    if !filename.starts_with("dielectric_") {
        return None;
    }

    let parts: Vec<&str> = filename.split('_').collect();
    let mut params = DielectricParams {
        a0: 0.0,
        v0: 0.0,
        n_rel: 0,
        lm_coupling: 0.0,
        n_modes: 0,
        pn_coupling: 0.0,
        pn_n_fock: 0,
        n_eig: 0,
        gamma: 0.0,
    };

    for part in parts.iter() {
        if part.starts_with("a0") {
            params.a0 = part[2..].parse::<f64>().unwrap_or(0.0);
        } else if part.starts_with("V0") {
            params.v0 = part[2..].parse::<f64>().unwrap_or(0.0);
        } else if part.starts_with("nrel") {
            params.n_rel = part[4..].parse::<usize>().unwrap_or(0);
        } else if part.starts_with("lmcoup") {
            params.lm_coupling = part[6..].parse::<f64>().unwrap_or(0.0);
        } else if part.starts_with("nmodes") {
            params.n_modes = part[6..].parse::<usize>().unwrap_or(0);
        } else if part.starts_with("pncoup") {
            params.pn_coupling = part[6..].parse::<f64>().unwrap_or(0.0);
        } else if part.starts_with("npn") {
            params.pn_n_fock = part[3..].parse::<usize>().unwrap_or(0);
        } else if part.starts_with("neig") {
            params.n_eig = part[4..].parse::<usize>().unwrap_or(0);
        } else if part.starts_with("gamma") {
            let gamma_str = part[5..].trim_end_matches(".dat");
            params.gamma = gamma_str.parse::<f64>().unwrap_or(0.0);
        }
    }

    Some(params)
}

#[macro_export]
macro_rules! plot_photon_freqs {
    ($chart:expr, $y_min:expr, $y_max:expr, $qs:expr) => {
        // Style: light‑gray, 1 pixel width
        let line_style = ShapeStyle {
            color: RGBColor(200, 200, 200).mix(0.8),
            filled: false,
            stroke_width: 1,
        };

        const C: f64 = 137.036;

        let freqs: Vec<f64> = $qs.iter().map(|q| q.magnitude() * C).collect();

        // Current x‑range of the chart (used to skip out‑of‑range lines)
        let x_range = $chart.x_range();

        for &freq in freqs.iter() {
            // Only draw if the frequency is inside the plotted ω‑range
            if freq >= x_range.start && freq <= x_range.end {
                // Two‑point line: (freq, y_min) → (freq, y_max)
                $chart.draw_series(LineSeries::new(
                    vec![(freq, $y_min), (freq, $y_max)],
                    line_style.clone(),
                ))?;
            }
        }
    };
}

/// Plots the dielectric function for different coupling strengths
pub fn plot_dielectric_function(
    dir_path: &str,                  // Directory containing dielectric function data files
    save_path: Option<&str>,         // Path to save the plot
    plot_type: DielectricPlotType,   // Which part of the dielectric function to plot
    omega_range: Option<(f64, f64)>, // Optional frequency range to plot
    coupling_values: &Array1<f64>,
    reference_params: Option<&DielectricParams>, // Optional reference parameters to match against
    zoomed: bool,
    q_values: Option<&Vec<Vector3D>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = false;
    let dark_mode = false;

    let mut background = &WHITE;
    let mut text_color = &BLACK;

    let mut ref_colors = vec![(0., 0., 0.), (255., 0., 0.), (0., 0., 255.)];

    if dark_mode {
        background = &BLACK;
        text_color = &WHITE;
        ref_colors = vec![(237., 110., 87.), (148., 33., 148.), (72., 159., 248.)];
    }

    // let colors = [RGBColor(0,0,0), RGBColor(255,0,0), RGBColor(0,0,255)];

    timed_println!("Plotting dielectric function from directory: {}", dir_path);
    // Get all dielectric function files in the directory
    let dir = Path::new(dir_path);
    let mut files_with_params = std::fs::read_dir(dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_file() && path.extension()?.to_str()? == "dat" {
                // Extract parameters from filename
                if let Some(params) = extract_params_from_filename(&path) {
                    // Check if this coupling value is EXACTLY in our list (with small tolerance)
                    let coupling_match = coupling_values
                        .iter()
                        .any(|&c| (c - params.lm_coupling).abs() < 1e-10);

                    // Check if other parameters match the reference parameters (if provided)
                    let params_match = if let Some(ref_params) = reference_params {
                        // Allow small tolerance for floating point comparisons
                        let a0_match = (params.a0 - ref_params.a0).abs() < 1e-10;
                        let v0_match = (params.v0 - ref_params.v0).abs() < 1e-10;
                        let n_rel_match = params.n_rel == ref_params.n_rel;
                        let n_modes_match = params.n_modes == ref_params.n_modes;
                        let pn_coupling_match =
                            (params.pn_coupling - ref_params.pn_coupling).abs() < 1e-10;
                        let pn_n_fock_match = params.pn_n_fock == ref_params.pn_n_fock;
                        let n_eig_match = params.n_eig == ref_params.n_eig;
                        let gamma_match = params.gamma == ref_params.gamma;

                        let all_match = a0_match
                            && v0_match
                            && n_rel_match
                            && n_modes_match
                            && pn_coupling_match
                            && pn_n_fock_match
                            && n_eig_match
                            && gamma_match;

                        if !all_match && verbose {
                            timed_println!(
                                "Parameter mismatch for file with coupling {:.4}: {}",
                                params.lm_coupling,
                                path.display()
                            );
                            if !a0_match {
                                timed_println!("  a0: {} vs {}", params.a0, ref_params.a0);
                            }
                            if !v0_match {
                                timed_println!("  v0: {} vs {}", params.v0, ref_params.v0);
                            }
                            if !n_rel_match {
                                timed_println!("  n_rel: {} vs {}", params.n_rel, ref_params.n_rel);
                            }
                            if !n_modes_match {
                                timed_println!(
                                    "  n_modes: {} vs {}",
                                    params.n_modes,
                                    ref_params.n_modes
                                );
                            }
                            if !pn_coupling_match {
                                timed_println!(
                                    "  pn_coupling: {} vs {}",
                                    params.pn_coupling,
                                    ref_params.pn_coupling
                                );
                            }
                            if !pn_n_fock_match {
                                timed_println!(
                                    "  pn_n_fock: {} vs {}",
                                    params.pn_n_fock,
                                    ref_params.pn_n_fock
                                );
                            }
                            if !n_eig_match {
                                timed_println!("  n_eig: {} vs {}", params.n_eig, ref_params.n_eig);
                            }
                            if !gamma_match {
                                timed_println!("  gamma: {} vs {}", params.gamma, ref_params.gamma);
                            }
                        }

                        all_match
                    } else {
                        // If no reference parameters provided, only match on coupling
                        true
                    };

                    if coupling_match && params_match {
                        timed_println!(
                            "Found matching file for coupling {:.4}: {}",
                            params.lm_coupling,
                            path.display()
                        );
                        return Some((path, params));
                    } else if verbose {
                        timed_println!(
                            "Skipping file with coupling {:.4} (not in requested list): {}",
                            params.lm_coupling,
                            path.display()
                        );
                    }
                }
                // else if let Some(coupling) = extract_coupling_from_filename(&path) {
                //     // Fall back to old format
                //     let coupling_match = coupling_values.iter().any(|&c| (c - coupling).abs() < 1e-10);
                //     if coupling_match {
                //         // Create a default params object with just the coupling
                //         let default_params = DielectricParams {
                //             a0: 0.0,
                //             v0: 0.0,
                //             n_rel: 0,
                //             lm_coupling: coupling,
                //             n_modes: 0,
                //             pn_coupling: 0.0,
                //             pn_n_fock: 0,
                //             n_eig: 0,
                //             gamma: 0.0,
                //         };
                //         timed_println!("Found matching file (old format) for coupling {:.4}: {}", coupling, path.display());
                //         return Some((path, default_params));
                //     } else if verbose {
                //         timed_println!("Skipping file with coupling {:.4} (old format, not in requested list): {}",
                //                      coupling, path.display());
                //     }
                // }
                else if verbose {
                    timed_println!(
                        "Skipping file (couldn't extract coupling): {}",
                        path.display()
                    );
                }
            }
            None
        })
        .collect::<Vec<_>>();

    // Print the requested coupling values for comparison
    timed_println!("Requested coupling values: {:?}", coupling_values);

    // Sort files by coupling strength
    files_with_params.sort_by(|(_, params_a), (_, params_b)| {
        params_a
            .lm_coupling
            .partial_cmp(&params_b.lm_coupling)
            .unwrap()
    });

    if files_with_params.is_empty() {
        return Err("No matching dielectric function data files found".into());
    }

    // Check if we found exactly the number of files we expected
    if files_with_params.len() != coupling_values.len() {
        timed_println!(
            "Warning: Found {} files but expected {} (one for each coupling value)",
            files_with_params.len(),
            coupling_values.len()
        );

        // Print what we found vs what we expected
        let found_couplings: Vec<f64> = files_with_params
            .iter()
            .map(|(_, params)| params.lm_coupling)
            .collect();
        timed_println!("Found couplings: {:?}", found_couplings);
        timed_println!("Expected couplings: {:?}", coupling_values);
    } else {
        timed_println!(
            "Found exactly {} matching dielectric function files as expected",
            files_with_params.len()
        );
    }

    // Load data from files
    let mut all_data: HashMap<PathBuf, (Vec<f64>, Vec<f64>, Vec<f64>, DielectricParams)> =
        HashMap::new();
    let mut omega_min = f64::INFINITY;
    let mut omega_max = f64::NEG_INFINITY;
    let mut eps_min: (f64, f64) = (f64::INFINITY, f64::INFINITY);
    let mut eps_max: (f64, f64) = (f64::NEG_INFINITY, f64::NEG_INFINITY);

    for (file_path, params) in &files_with_params {
        let (omegas, re_eps, im_eps) = load_dielectric_data(file_path)?;

        // Update min/max values
        if let Some(&min_omega) = omegas.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            omega_min = omega_min.min(min_omega);
        }
        if let Some(&max_omega) = omegas.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            omega_max = omega_max.max(max_omega);
        }

        let chart_data = (
            Array1::from_vec(re_eps.clone()),
            Array1::from_vec(im_eps.clone()),
        );

        let mean = (chart_data.0.mean().unwrap(), chart_data.1.mean().unwrap());
        let mean2 = (
            (&chart_data.0 * &chart_data.0).mean().unwrap(),
            (&chart_data.1 * &chart_data.1).mean().unwrap(),
        );

        let sigma = (
            (mean2.0 - mean.0 * mean.0).sqrt(),
            (mean2.1 - mean.1 * mean.1).sqrt(),
        );

        let factor = 0.5;
        eps_min = (
            eps_min.0.min(mean.0 - factor * sigma.0),
            eps_min.1.min(mean.1 - 3. * factor * sigma.1),
        );
        eps_max = (
            eps_max.0.max(mean.0 + factor * sigma.0),
            eps_max.1.max(mean.1 + 3. * factor * sigma.1),
        );

        all_data.insert(file_path.clone(), (omegas, re_eps, im_eps, params.clone()));
    }

    // Apply omega range if specified
    if let Some((min, max)) = omega_range {
        omega_min = min;
        omega_max = max;
    }

    // Create the plot
    let resolution = if zoomed {
        (1600, 1000)
    } else {
        (1600 * 2, 1000 * 2)
    };
    let save_path = save_path.unwrap_or("dielectric_function.svg");
    let root = SVGBackend::new(save_path, resolution).into_drawing_area(); // Increased plot size
    root.fill(background)?;

    // Split the drawing area for multiple plots if needed
    let areas = if plot_type == DielectricPlotType::Both {
        let (top, bottom) = root.split_vertically(500); // Adjusted for new dimensions
        vec![top, bottom]
    } else {
        vec![root.clone()]
    };

    // Get sorted couplings with file paths
    let mut couplings: Vec<(PathBuf, f64)> = all_data
        .iter()
        .map(|(file_path, (_, _, _, params))| (file_path.clone(), params.lm_coupling))
        .collect();
    couplings.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

    // Extract common parameters for the title
    let mut common_params = None;
    for (_, (_, _, _, params)) in all_data.iter() {
        if common_params.is_none() {
            common_params = Some(params.clone());
        }
    }

    // Plot title with parameters
    let title_base = match plot_type {
        DielectricPlotType::Real => "Real Part of Dielectric Function",
        DielectricPlotType::Imaginary => "Imaginary Part of Dielectric Function",
        DielectricPlotType::Both => "Dielectric Function",
    };

    let title = if let Some(params) = common_params {
        if params.n_modes > 0 {
            format!(
                "{} (a₀={:.1}, V₀={:.3}, {} modes)",
                title_base, params.a0, params.v0, params.n_modes
            )
        } else {
            title_base.to_string()
        }
    } else {
        title_base.to_string()
    };

    // Plot the real part
    if plot_type == DielectricPlotType::Real || plot_type == DielectricPlotType::Both {
        // Symlog transformation with threshold
        let threshold = 1000.0;
        let symlog = |x: f64| -> f64 {
            if x.abs() < threshold {
                x
            } else if x > 0.0 {
                threshold * (x / threshold).ln() + threshold
            } else {
                -threshold * ((-x) / threshold).ln() - threshold
            }
        };

        // Inverse symlog for axis labels
        let inv_symlog = |y: f64| -> f64 {
            if y.abs() < threshold {
                y
            } else if y > 0.0 {
                threshold * ((y - threshold) / threshold).exp()
            } else {
                -threshold * (((-y) - threshold) / threshold).exp()
            }
        };

        // Compute symlog-transformed range for all data
        let mut transformed_min = f64::INFINITY;
        let mut transformed_max = f64::NEG_INFINITY;

        for (_, (_, re_eps, _, _)) in all_data.iter() {
            for &eps in re_eps.iter() {
                let transformed = symlog(eps);
                transformed_min = transformed_min.min(transformed);
                transformed_max = transformed_max.max(transformed);
            }
        }

        // Add some padding to the transformed range
        let padding = (transformed_max - transformed_min) * 0.05;
        transformed_min -= padding;
        transformed_max += padding;

        let area_index = if plot_type == DielectricPlotType::Both {
            0
        } else {
            0
        };
        let mut chart = ChartBuilder::on(&areas[area_index])
            .caption(
                format!(
                    "{} - Real Part (symlog scale, threshold={})",
                    title, threshold
                ),
                ("sans-serif", 30).into_font().color(text_color),
            )
            .margin(10)
            .margin_top(20)
            .margin_bottom(10)
            .x_label_area_size(40)
            .y_label_area_size(70)
            .build_cartesian_2d(omega_min..omega_max, transformed_min..transformed_max)?;

        chart
            .configure_mesh()
            .x_desc("Frequency (a.u.)")
            .y_desc("Re[ε(ω)]")
            .axis_desc_style(("sans-serif", 20).into_font().color(text_color))
            .label_style(("sans-serif", 18).into_font().color(text_color))
            .y_label_formatter(&|y| {
                // Inverse symlog to show original values on axis
                let original = inv_symlog(*y);
                if original.abs() < 0.01 {
                    format!("{:.3}", original)
                } else if original.abs() < 1.0 {
                    format!("{:.2}", original)
                } else if original.abs() < 10.0 {
                    format!("{:.1}", original)
                } else {
                    format!("{:.0}", original)
                }
            })
            .disable_mesh() // Remove fine grid lines
            .draw()?;

        if let Some(qs) = q_values {
            plot_photon_freqs!(&mut chart, transformed_min, transformed_max, qs);
        }

        for (i, (key, coupling)) in couplings.iter().enumerate() {
            let (omegas, re_eps, _, params) = &all_data[key];

            // Create a color gradient from blue to red
            let color = interpolate_colormap(&ref_colors, i as f64 / (couplings.len() - 1) as f64);
            // let color = colors[i];

            // Filter data points within the omega range and apply symlog transformation
            let points: Vec<(f64, f64)> = omegas
                .iter()
                .zip(re_eps.iter())
                .filter(|(omega, _)| omega >= &&omega_min && omega <= &&omega_max)
                .map(|(&omega, &eps)| (omega, symlog(eps)))
                .collect();

            // Create a more detailed label if we have the parameters
            let label = if params.n_modes > 0 {
                format!("g = {:.3}, {} modes", coupling, params.n_modes)
            } else {
                format!("g = {:.3}", coupling)
            };

            chart
                .draw_series(LineSeries::new(
                    points,
                    ShapeStyle {
                        color: color.into(),
                        filled: true,
                        stroke_width: 6, // Increased line thickness
                    },
                ))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }

        // // Add legend
        // chart.configure_series_labels()
        //     .background_style(&WHITE.mix(0.9)) // More opaque background
        //     .border_style(&BLACK)
        //     .label_font(("sans-serif", 16)) // Slightly larger font
        //     .position(SeriesLabelPosition::UpperRight)
        //     .margin(15) // Add some margin around the legend text
        //     .draw()?;
    }

    // Plot the imaginary part
    if plot_type == DielectricPlotType::Imaginary || plot_type == DielectricPlotType::Both {
        // Change to semilog plot
        let mut e_min: f64 = f64::INFINITY;
        let mut e_max = f64::NEG_INFINITY;

        for (_, (omegas, _, im_eps, _)) in all_data.iter() {
            for (&eps, _) in im_eps
                .iter()
                .zip(omegas.iter())
                .filter(|(_, omega)| *omega > &omega_min && *omega < &omega_max)
            {
                e_max = e_max.max(eps.abs());
                e_min = e_min.min(eps.abs());
            }
        }

        println!("max: {}, min: {}", e_max, e_min);

        let area_index = if plot_type == DielectricPlotType::Both {
            1
        } else {
            0
        };
        let mut chart = ChartBuilder::on(&areas[area_index])
            .caption(
                format!("{} - Imaginary Part", title),
                ("sans-serif", 30).into_font().color(text_color),
            )
            .margin(10)
            .margin_top(20)
            .margin_bottom(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            // .build_cartesian_2d(omega_min..omega_max, 2.0*eps_min.1..eps_max.0 * 0.05)?;
            // .build_cartesian_2d(omega_min..omega_max, (e_min..e_max).log_scale())?;
            .build_cartesian_2d(omega_min..omega_max, e_min..e_max)?;

        println!("Built mesh");

        chart
            .configure_mesh()
            .x_desc("Frequency (a.u.)")
            .y_desc("Im[ε(ω)]")
            .axis_desc_style(("sans-serif", 20).into_font().color(text_color))
            .axis_style(text_color)
            .label_style(("sans-serif", 18).into_font().color(text_color))
            .disable_mesh() // Remove fine grid lines
            .draw()?;

        println!("Configured mesh");

        // Draw each coupling strength with a different color
        for (i, (key, coupling)) in couplings.iter().enumerate().rev() {
            let (omegas, _, im_eps, params) = &all_data[key];

            let color = interpolate_colormap(&ref_colors, i as f64 / (couplings.len() - 1) as f64);
            // let color = colors[i];

            // Filter data points within the omega range
            let points: Vec<(f64, f64)> = omegas
                .iter()
                .zip(im_eps.iter())
                .filter(|(omega, _)| omega >= &&omega_min && omega <= &&omega_max)
                // .map(|(&omega, &eps)| (omega, eps))
                .map(|(&omega, &eps)| (omega, eps.abs()))
                .collect();

            // Create a more detailed label if we have the parameters
            let label = if params.n_modes > 0 {
                format!(
                    "g = {:e}, {} modes",
                    coupling * (params.n_modes as f64).sqrt(),
                    params.n_modes
                )
            } else {
                format!("g = {:e}", coupling * (params.n_modes as f64).sqrt())
            };

            chart
                .draw_series(LineSeries::new(
                    points,
                    ShapeStyle {
                        color: color.into(),
                        filled: true,
                        stroke_width: 8, // Increased line thickness
                    },
                ))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }

        println!("Plotted Couplings");

        if let Some(qs) = q_values {
            plot_photon_freqs!(&mut chart, e_min, e_max, qs);
        }

        // if plot_type == DielectricPlotType::Imaginary && !zoomed{
        //     // Add legend
        //     chart.configure_series_labels()
        //         .background_style(&WHITE.mix(0.9)) // More opaque background
        //         .border_style(&BLACK)
        //         .label_font(("sans-serif", 16)) // Slightly larger font
        //         .position(SeriesLabelPosition::LowerRight)
        //         .margin(15) // Add some margin around the legend text
        //         .draw()?;
        // }
    }

    // Present the final plot
    root.present()?;
    info!("Dielectric function plot saved to {}", save_path);

    Ok(())
}

/// Helper function to extract coupling strength from filename
fn _extract_coupling_from_filename(path: &Path) -> Option<f64> {
    let filename = path.file_name()?.to_str()?;
    let parts: Vec<&str> = filename.split('_').collect();

    // Try the new format first
    for part in parts.iter() {
        if part.starts_with("lmcoup") {
            return part[6..].parse::<f64>().ok();
        }
    }

    // Fall back to the old format
    for (i, part) in parts.iter().enumerate() {
        if *part == "coupling" && i + 1 < parts.len() {
            return parts[i + 1].trim_end_matches(".dat").parse::<f64>().ok();
        }
    }

    None
}

/// Helper function to load dielectric function data from a file
pub fn load_dielectric_data(
    path: &Path,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
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

        let values: Vec<f64> = line
            .split_whitespace()
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
    k_points: &Vec<Vec<f64>>,             // k-points along the path
    sym_points: Vec<Vec<f64>>,            // High-symmetry points
    sym_labels: Vec<String>,              // Labels for high-symmetry points
    eigvals_list_full: &Vec<Array1<f64>>, // Eigenvalues at each k-point
    n_bands: usize,                       // Number of bands to plot
    a0: f64,                              // Lattice constant
    save_path: Option<&str>,              // Path to save the plot
    _show: bool,                          // Whether to show the plot (unused)
    v0: f64,                              // Potential strength
    coupling: f64,                        // Light-matter coupling
) {
    let dark_mode = false;

    let mut background = &WHITE;
    let mut text_color = &BLACK;

    if dark_mode {
        background = &BLACK;
        text_color = &WHITE;
    }

    if let Some(path) = save_path {
        info!("Saving band structure plot to {}", path);

        // Create a larger drawing area with better resolution
        let root = BitMapBackend::new(path, (1200, 900)).into_drawing_area();
        root.fill(background).unwrap();

        // Create x-axis for the plot (path length)
        let mut x = vec![0.0; k_points.len()];
        for i in 1..k_points.len() {
            let dx = k_points[i][0] - k_points[i - 1][0];
            let dy = k_points[i][1] - k_points[i - 1][1];
            x[i] = x[i - 1] + (dx * dx + dy * dy).sqrt();
        }

        let dx = k_points[1][0] - k_points[0][0];

        // Define the high-symmetry points
        let points_per_segment = k_points.len() / 3;
        let n_sym = sym_points.len();

        // Calculate positions of symmetry points on the x-axis
        let mut sym_positions: Vec<f64> = sym_points
            .iter()
            .enumerate()
            .map(|(ind, _)| x[ind * points_per_segment])
            .collect();
        sym_positions.push(x[x.len() - 1]);

        // Find min/max energy values with some padding
        let mut min_energy = f64::INFINITY;
        let mut max_energy = f64::NEG_INFINITY;

        // Limit the number of bands to plot
        let n_bands = n_bands.min(eigvals_list_full[0].len());

        // Extract the bands to plot
        let eigvals_list: &Vec<Array1<f64>> = &eigvals_list_full
            .iter()
            .map(|eigvals| eigvals.slice(s![0..n_bands - 1]).to_owned())
            .collect();

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
        let title = format!(
            "Exciton-Polariton Band Structure (a₀={:.1}, V₀={:.1}, g={:.3})",
            a0, v0, coupling
        );

        // Set up chart with better styling
        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 30).into_font().color(text_color))
            .margin(10)
            .margin_top(20)
            .margin_bottom(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x[0]..x[x.len() - 1], min_energy..max_energy)
            .unwrap();

        // Configure mesh with custom x-ticks for high-symmetry points
        chart
            .configure_mesh()
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
            .axis_desc_style(("sans-serif", 20).into_font().color(text_color))
            .label_style(("sans-serif", 18).into_font().color(text_color))
            .disable_mesh()
            .draw()
            .unwrap();

        // Draw vertical lines at high-symmetry points
        let line_style = ShapeStyle {
            color: text_color.mix(0.6),
            filled: false,
            stroke_width: 3,
        };

        for &x_pos in &sym_positions {
            chart
                .draw_series(LineSeries::new(
                    vec![(x_pos, min_energy), (x_pos, max_energy)],
                    line_style,
                ))
                .unwrap();
        }

        // Draw each band with the same color (blue)
        let num_bands = (&eigvals_list[0]).len();
        timed_println!("Number of bands: {}", num_bands);

        for band_idx in 0..num_bands {
            let color = BLACK;
            let band_points: Vec<(f64, f64)> = x
                .iter()
                .zip(eigvals_list.iter())
                .map(|(&x_val, eigvals)| (x_val, eigvals[band_idx]))
                .collect();

            chart
                .draw_series(LineSeries::new(
                    band_points,
                    ShapeStyle {
                        color: color.mix(1.0),
                        filled: false,
                        stroke_width: 3,
                    },
                ))
                .unwrap();
        }

        // Present the final plot
        root.present().unwrap();
        info!("Band structure plot saved to {}", path);
    }
}

fn interpolate_colormap(colors: &[(f64, f64, f64)], t: f64) -> RGBColor {
    // Handle edge cases
    if colors.is_empty() {
        return RGBColor(0, 0, 0);
    }
    if colors.len() == 1 {
        return RGBColor(colors[0].0 as u8, colors[0].1 as u8, colors[0].2 as u8);
    }

    // Clamp t to [0.0, 1.0]
    let t = t.clamp(0.0, 1.0);

    // Calculate which segment we're in
    let n_segments = colors.len() - 1;
    let segment_length = 1.0 / n_segments as f64;
    let segment_index = (t / segment_length).floor() as usize;

    // Handle the case where t = 1.0 exactly
    if segment_index >= n_segments {
        return RGBColor(
            colors[colors.len() - 1].0 as u8,
            colors[colors.len() - 1].1 as u8,
            colors[colors.len() - 1].2 as u8,
        );
    }

    // Calculate local interpolation parameter within the segment
    let local_t = (t - segment_index as f64 * segment_length) / segment_length;

    // Get the two colors to interpolate between
    let color1 = colors[segment_index];
    let color2 = colors[segment_index + 1];

    // Linear interpolation
    let r = color1.0 + (color2.0 - color1.0) * local_t;
    let g = color1.1 + (color2.1 - color1.1) * local_t;
    let b = color1.2 + (color2.2 - color1.2) * local_t;

    RGBColor(r as u8, g as u8, b as u8)
}
