use log::info;
use ndarray::Array1;
use plotters::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::data::{DielectricParams, extract_params_from_filename, load_dielectric_data};
use crate::{Vector3D, plot_photon_freqs, timed_println};

#[derive(PartialEq)]
pub enum DielectricPlotType {
    Real,
    Imaginary,
    Both,
}

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
                        .any(|&c| (c - params.lm_coupling).abs() < 1e-5);

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
                        let gamma_match = (params.gamma - ref_params.gamma).abs() < 1e-12;

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
            plot_photon_freqs!(chart, transformed_min, transformed_max, qs);
        }

        for (i, (key, coupling)) in couplings.iter().enumerate() {
            let (omegas, re_eps, _, params) = &all_data[key];

            // Create a color gradient from blue to red
            let color = interpolate_colormap(
                &ref_colors,
                if couplings.len() > 1 {
                    if couplings.len() > 1 {
                        i as f64 / (couplings.len() - 1) as f64
                    } else {
                        0.0
                    }
                } else {
                    0.0
                },
            );
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
            .build_cartesian_2d(omega_min..omega_max, (e_min..e_max).log_scale())?;
        // .build_cartesian_2d(omega_min..omega_max, e_min..e_max)?;

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

            let color = interpolate_colormap(
                &ref_colors,
                if couplings.len() > 1 {
                    if couplings.len() > 1 {
                        i as f64 / (couplings.len() - 1) as f64
                    } else {
                        0.0
                    }
                } else {
                    0.0
                },
            );
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
            plot_photon_freqs!(chart, e_min, e_max, qs);
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
