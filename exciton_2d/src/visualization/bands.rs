use log::info;
use ndarray::Array1;
use plotters::prelude::*;

pub fn plot_band_structure(
    k_points: &[Vec<f64>],
    eigvals_list: &[Array1<f64>],
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
        let x = sim_common::kpath::path_distances(k_points);

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
            .build_cartesian_2d(x[0]..x[x.len() - 1], min_energy..max_energy)
            .unwrap();

        // Configure mesh with custom x-ticks for high-symmetry points
        chart
            .configure_mesh()
            .x_labels(4)
            .x_label_formatter(&|v| {
                if (v - &x_gamma1).abs() < 1e-6 {
                    "Γ".to_string()
                } else if (v - &x_k).abs() < 1e-6 {
                    "X".to_string()
                } else if (v - &x_m).abs() < 1e-6 {
                    "M".to_string()
                } else if (v - &x_gamma2).abs() < 1e-6 {
                    "Γ".to_string()
                } else {
                    "".to_string()
                }
            })
            .y_desc("Energy")
            .draw()
            .unwrap();

        // Draw vertical lines at high-symmetry points
        chart
            .draw_series(vec![
                PathElement::new(
                    vec![(x_gamma1, min_energy), (x_gamma1, max_energy)],
                    BLACK.mix(0.5),
                ),
                PathElement::new(vec![(x_k, min_energy), (x_k, max_energy)], BLACK.mix(0.5)),
                PathElement::new(vec![(x_m, min_energy), (x_m, max_energy)], BLACK.mix(0.5)),
                PathElement::new(
                    vec![(x_gamma2, min_energy), (x_gamma2, max_energy)],
                    BLACK.mix(0.5),
                ),
            ])
            .unwrap();

        // Draw each band with colors based on parity values
        for band_idx in 0..eigvals_list[0].len() {
            for k_idx in 0..x.len() - 1 {
                // Get parity value for this segment
                let color = BLACK;

                // Plot the segment with the appropriate color
                chart
                    .draw_series(LineSeries::new(
                        vec![
                            (x[k_idx], eigvals_list[k_idx][band_idx]),
                            (x[k_idx + 1], eigvals_list[k_idx + 1][band_idx]),
                        ],
                        color.mix(0.8),
                    ))
                    .unwrap();
            }
        }

        root.present().unwrap();
    }
}
