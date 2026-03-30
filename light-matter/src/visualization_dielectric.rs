//! Visualization utilities for the dielectric function.
//!
//! Two helper functions are provided:
//! 1. `compute_weight_fields` – given the eigen‑state contributions (`pp_vals`,
//!    `x_pp`) and the per‑eigenstate composition data (`photonic_chars`,
//!    `phonon_numbers`), it returns two scalar fields defined on the frequency
//!    grid `omegas`.  Each field is the **weighted average** of the corresponding
//!    composition quantity, where the weight for eigenstate *i* at frequency
//!    `ω` is the magnitude of its contribution to the dielectric function:
//!    `|x_pp[i]^2 / (ω - ΔE_i + iγ)|`.
//! 2. `plot_dielectric_with_color` – draws a single line plot of the real part
//!    of the dielectric function where the colour of each segment is taken from
//!    a supplied weight field (photonic character or phonon number).  The output
//!    is an SVG file.

use ndarray::Array1;
use ndarray_linalg::c64;
use plotters::prelude::*;

/// Compute the weighted‑average colour fields for the dielectric function.
///
/// * `omegas` – frequency grid.
/// * `pp_vals` – eigen energies (absolute, not shifted).
/// * `x_pp` – transition dipole strengths for each eigenstate.
/// * `photonic_chars` – photonic character for each eigenstate.
/// * `phonon_numbers` – average phonon number for each eigenstate.
/// * `gamma` – broadening parameter used in the dielectric denominator.
///
/// Returns `(photonic_curve, phonon_curve)`, each an `Array1<f64>` of the same
/// length as `omegas`.
pub fn compute_weight_fields(
    omegas: &Array1<f64>,
    pp_vals: &Array1<f64>,
    x_pp: &Array1<f64>,
    photonic_chars: &[f64],
    phonon_numbers: &[f64],
    gamma: f64,
) -> (Array1<f64>, Array1<f64>) {
    let n_states = pp_vals.len();
    let mut photonic_curve = Array1::<f64>::zeros(omegas.len());
    let mut phonon_curve = Array1::<f64>::zeros(omegas.len());

    for (w_idx, &omega) in omegas.iter().enumerate() {
        // accumulate numerator and denominator for each weight field
        let mut w_sum = 0.0f64;
        let mut phot_sum = 0.0f64;
        let mut phon_sum = 0.0f64;
        for i in 0..n_states {
            let delta_e = pp_vals[i] - pp_vals[0];
            let contrib = x_pp[i] * x_pp[i] / (omega - delta_e + c64::i() * gamma);
            let w_i = contrib.norm(); // magnitude of complex contribution
            w_sum += w_i;
            phot_sum += w_i * photonic_chars[i];
            phon_sum += w_i * phonon_numbers[i];
        }
        if w_sum != 0.0 {
            photonic_curve[w_idx] = phot_sum / w_sum;
            phonon_curve[w_idx] = phon_sum / w_sum;
        }
    }

    (photonic_curve, phonon_curve)
}

/// Plot the real part of the dielectric function with colour representing a
/// weight field (e.g., photonic character or phonon number).
///
/// The plot is saved as an SVG at `save_path`.
pub fn plot_dielectric_with_color(
    omegas: &Array1<f64>,
    dielectric_real: &Array1<f64>,
    weight: &Array1<f64>,
    title: &str,
    save_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Determine colour scaling
    let min_w = weight.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_w = weight.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_w - min_w;

    let root = BitMapBackend::new(save_path, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_min = omegas[0];
    let x_max = omegas[omegas.len() - 1];
    let y_min = dielectric_real.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = dielectric_real.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 24))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    // Draw coloured segments
    for i in 0..(omegas.len() - 1) {
        let x0 = omegas[i];
        let x1 = omegas[i + 1];
        let y0 = dielectric_real[i];
        let y1 = dielectric_real[i + 1];
        // average weight for the segment
        let w = (weight[i] + weight[i + 1]) / 2.0;
        let ratio = if range.abs() < 1e-12 { 0.5 } else { (w - min_w) / range };
        // linear interpolation between blue and red
        let r = (ratio * 255.0) as u8;
        let b = ((1.0 - ratio) * 255.0) as u8;
        let colour = RGBColor(r, 0, b);
        chart
            .draw_series(LineSeries::new(vec![(x0, y0), (x1, y1)], &colour))?
            .label(format!("{:.3}", w))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &colour));
    }

    // Optional legend (show colour scale)
    chart.configure_series_labels().border_style(&BLACK).draw()?;

    root.present()?;
    Ok(())
}
