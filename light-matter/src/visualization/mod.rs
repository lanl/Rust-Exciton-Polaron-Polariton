pub mod bands;
pub mod data;
pub mod dielectric;
pub mod stacked;

pub use bands::plot_band_structure;
pub use data::{DielectricParams, extract_params_from_filename, load_dielectric_data};
pub use dielectric::{DielectricPlotType, plot_dielectric_function};
pub use stacked::{
    compute_weight_fields, generate_stacked_plots, plot_dielectric_stacked_phonon,
    plot_dielectric_stacked_photonic,
};

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
