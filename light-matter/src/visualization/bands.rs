use log::info;
use ndarray::{Array1, s};
use plotters::prelude::*;

use crate::timed_println;

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

    if k_points.len() < 2 || eigvals_list_full.is_empty() || eigvals_list_full[0].is_empty() {
        return;
    }

    if let Some(path) = save_path {
        info!("Saving band structure plot to {}", path);

        // Create a larger drawing area with better resolution
        let root = BitMapBackend::new(path, (1200, 900)).into_drawing_area();
        root.fill(background).unwrap();

        // Create x-axis for the plot (path length)
        let x = sim_common::kpath::path_distances(k_points);

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
            .map(|eigvals| eigvals.slice(s![0..n_bands]).to_owned())
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
