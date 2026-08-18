use log::info;
use ndarray::Array1;
use plotters::prelude::*;

pub fn plot_spectrum(eigvals: &Array1<f64>, k: &[f64], save_path: Option<&str>) {
    // This is a placeholder for the spectrum plotting functionality

    if let Some(path) = save_path {
        info!("Saving spectrum plot to {}", path);

        let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("Exciton energy spectrum at K = {:?}", k),
                ("sans-serif", 20),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0.0..eigvals.len() as f64,
                eigvals.fold(f64::INFINITY, |a, &b| a.min(b))
                    ..eigvals.fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            )
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("State index")
            .y_desc("Energy")
            .draw()
            .unwrap();

        // Draw points
        chart
            .draw_series(
                (0..eigvals.len()).map(|i| Circle::new((i as f64, eigvals[i]), 3, BLUE.filled())),
            )
            .unwrap();

        root.present().unwrap();
    }
}
