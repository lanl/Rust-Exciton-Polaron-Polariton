//! Visualization utilities for the dielectric function.
//!
//! Three helper functions are provided:
//! 1. `compute_weight_fields` – given the eigen‑state contributions (`pp_vals`,
//!    `x_pp`) and the per‑eigenstate composition data (`photonic_chars`,
//!    `phonon_numbers`), it returns two scalar fields defined on the frequency
//!    grid `omegas`.  Each field is the **weighted average** of the corresponding
//!    composition quantity, where the weight for eigenstate *i* at frequency
//!    `ω` is the magnitude of its contribution to the dielectric function:
//!    `|x_pp[i]^2 / (ω - ΔE_i + iγ)|`.
//! 2. `plot_dielectric_stacked_photonic` – creates a two-layer stacked area plot
//!    showing the photonic vs non-photonic contributions to Im[ε(ω)].
//! 3. `plot_dielectric_stacked_phonon` – creates a multi-layer stacked area plot
//!    showing contributions from each phonon Fock state (0, 1, 2, ..., n_fock).

use crate::plot_photon_freqs;
use crate::utils::Vector3D;
use ndarray::Array1;
use ndarray_linalg::c64;
use plotters::prelude::*;
use std::f64::consts::PI;

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
    ground_state_index: usize,
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
            if i == ground_state_index {
                continue;
            }

            let delta_e = pp_vals[i] - pp_vals[ground_state_index];
            let contrib = x_pp[i] * x_pp[i] / (omega - delta_e + c64::i() * gamma);
            let w_i = contrib.norm();

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

// /// Plot the real part of the dielectric function with colour representing a
// /// weight field (e.g., photonic character or phonon number).
// ///
// /// The plot is saved as an SVG at `save_path`.
// /// Plot the dielectric function with colour representing a weight field.
// ///
// /// This function mirrors the signature of `visualization::plot_dielectric_function`
// /// (directory handling, multiple coupling files, zoom, optional q‑values, etc.)
// /// but adds a `weight` array that is used to colour each line segment.
// /// The `weight` array must have the same length as the frequency grid (`omegas`).
// pub fn plot_dielectric_with_color(
//     dir_path: &str,                // Directory containing dielectric function data files
//     save_path: Option<&str>,       // Path to save the plot
//     plot_type: DielectricPlotType, // Which part of the dielectric function to plot
//     omega_range: Option<(f64, f64)>, // Optional frequency range to plot
//     coupling_values: &Array1<f64>,
//     reference_params: Option<&DielectricParams>, // Optional reference parameters to match against
//     zoomed: bool,
//     q_values: Option<&Vec<Vector3D>>,
//     weight: &Array1<f64>, // Weight field used for colour mapping (e.g., photonic character)
// ) -> Result<(), Box<dyn std::error::Error>> {
//     // The implementation mirrors `visualization::plot_dielectric_function` but
//     // uses the provided `weight` array to colour each segment.

//     // Load matching dielectric files (same logic as in plot_dielectric_function)
//     let verbose = false;
//     let dark_mode = false;
//     let mut background = &WHITE;
//     let mut text_color = &BLACK;
//     if dark_mode { background = &BLACK; text_color = &WHITE; }

//     // Collect files (reuse the same filtering logic as plot_dielectric_function)
//     let dir = std::path::Path::new(dir_path);
//     let mut files_with_params = std::fs::read_dir(dir)?
//         .filter_map(|entry| {
//             let entry = entry.ok()?;
//             let path = entry.path();
//             if path.is_file() && path.extension()?.to_str()? == "dat" {
//                 if let Some(params) = extract_params_from_filename(&path) {
//                     let coupling_match = coupling_values.iter().any(|&c| (c - params.lm_coupling).abs() < 1e-10);
//                     let params_match = if let Some(ref_params) = reference_params {
//                         (params.a0 - ref_params.a0).abs() < 1e-10 &&
//                         (params.v0 - ref_params.v0).abs() < 1e-10 &&
//                         params.n_rel == ref_params.n_rel &&
//                         params.n_modes == ref_params.n_modes &&
//                         (params.pn_coupling - ref_params.pn_coupling).abs() < 1e-10 &&
//                         params.pn_n_fock == ref_params.pn_n_fock &&
//                         params.n_eig == ref_params.n_eig &&
//                         (params.gamma - ref_params.gamma).abs() < 1e-10
//                     } else { true };
//                     if coupling_match && params_match {
//                         return Some((path, params));
//                     }
//                 }
//             }
//             None
//         })
//         .collect::<Vec<_>>();

//     if files_with_params.is_empty() {
//         return Err("No matching dielectric function data files found".into());
//     }

//     // Load the first matching file (for simplicity – weight is assumed to correspond to it)
//     let (file_path, params) = &files_with_params[0];
//     let (omegas_vec, re_eps, im_eps) = load_dielectric_data(file_path)?;
//     let omegas = Array1::from_vec(omegas_vec.clone());
//     let dielectric = match plot_type {
//         DielectricPlotType::Real => Array1::from_vec(re_eps.clone()),
//         DielectricPlotType::Imaginary => Array1::from_vec(im_eps.clone()),
//         DielectricPlotType::Both => Array1::from_vec(re_eps.clone()), // only real part for colour plot
//     };

//     // Determine colour scaling from the supplied weight array
//     let min_w = weight.iter().cloned().fold(f64::INFINITY, f64::min);
//     let max_w = weight.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
//     let range = max_w - min_w;

//     // Plot dimensions similar to plot_dielectric_function
//     let resolution = if zoomed {(1600, 1000)} else {(1600*2, 1000*2)};
//     let save_path = save_path.unwrap_or("dielectric_weighted.svg");
//     let root = SVGBackend::new(save_path, resolution).into_drawing_area();
//     root.fill(background)?;

//     let x_min = omegas[0];
//     let x_max = omegas[omegas.len() - 1];
//     let y_min = dielectric.iter().cloned().fold(f64::INFINITY, f64::min);
//     let y_max = dielectric.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

//     let mut chart = ChartBuilder::on(&root)
//         .caption(format!("{} (weight coloured)", match plot_type {
//             DielectricPlotType::Real => "Real Part",
//             DielectricPlotType::Imaginary => "Imaginary Part",
//             DielectricPlotType::Both => "Dielectric Function",
//         }), ("sans-serif", 24))
//         .margin(15)
//         .x_label_area_size(40)
//         .y_label_area_size(50)
//         .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

//     chart.configure_mesh().draw()?;

//     // Draw stacked area segments based on weight. Each segment is rendered as an
//     // AreaSeries that fills the region between the curve segment and the
//     // baseline (y = 0). The colour of each segment is determined by the weight
//     // value, producing a stacked‑area visualisation.
//     for i in 0..(omegas.len() - 1) {
//         let x0 = omegas[i];
//         let x1 = omegas[i + 1];
//         let y0 = dielectric[i];
//         let y1 = dielectric[i + 1];
//         let w = (weight[i] + weight[i + 1]) / 2.0;
//         let ratio = if range.abs() < 1e-12 { 0.5 } else { (w - min_w) / range };
//         let r = (ratio * 255.0) as u8;
//         let b = ((1.0 - ratio) * 255.0) as u8;
//         let colour = RGBColor(r, 0, b);
//         // AreaSeries expects an iterator of (x, y) points and a baseline value.
//         // We use a baseline of 0.0 to fill down to the x‑axis.
//         chart.draw_series(AreaSeries::new(vec![(x0, y0), (x1, y1)], 0.0, &colour))?;
//     }

//     // Draw photon frequencies if provided (reuse macro from visualization.rs)
//     if let Some(qs) = q_values {
//         // Re‑use the macro by creating a temporary chart reference
//         // Note: the macro expects a mutable chart reference; we can call it directly.
//         plot_photon_freqs!(chart, y_min, y_max, qs);
//     }

//     root.present()?;
//     Ok(())
// }

/// Plot the imaginary part of the dielectric function as a two-layer stacked area
/// showing photonic (red) vs non-photonic (blue) contributions.
///
/// At each frequency ω, the plot shows:
/// - Bottom layer (blue): Im[ε(ω)] × (1 - photonic_character(ω))
/// - Top layer (red): Im[ε(ω)] × photonic_character(ω)
/// where photonic_character(ω) is the weighted average photonic character.
///
/// This requires the raw eigenstate data (pp_vals, x_pp, photonic_chars) to compute
/// the weighted contributions at each frequency.
pub fn plot_dielectric_stacked_photonic(
    omegas: &Array1<f64>,
    pp_vals: &Array1<f64>,
    x_pp: &Array1<f64>,
    photonic_chars: &[f64],
    ground_state_index: usize,
    gamma: f64,
    save_path: &str,
    zoomed: bool,
    omega_range: Option<(f64, f64)>,
    q_values: Option<&Vec<Vector3D>>,
    zero_coupling_imag: Option<&Array1<f64>>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Compute the imaginary part of the dielectric function at each frequency
    let ground_state_energy = pp_vals[ground_state_index];
    let mut im_eps = Array1::<f64>::zeros(omegas.len());
    let mut photonic_weighted = Array1::<f64>::zeros(omegas.len());

    // Debug: Find states with highest contributions to dielectric function
    // Calculate total contribution strength for each state across all frequencies
    let mut state_contributions: Vec<(usize, f64, f64, f64)> = Vec::new(); // (index, energy, photonic_char, total_contribution)
    for i in 0..pp_vals.len() {
        if i == ground_state_index {
            continue;
        }
        let delta_e = pp_vals[i] - ground_state_energy;
        let x_sq = x_pp[i] * x_pp[i];
        // Sum |Im[contrib]| across frequency range of interest (0.17-0.18)
        let mut total_contrib = 0.0;
        for &omega in omegas.iter() {
            if omega >= 0.17 && omega <= 0.18 {
                let contrib = x_sq / (omega - delta_e + c64::i() * gamma);
                total_contrib += contrib.im.abs();
            }
        }
        state_contributions.push((i, delta_e, photonic_chars[i], total_contrib));
    }

    // Sort by total contribution (descending)
    state_contributions.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    println!("Ground state index: {}", ground_state_index);
    println!("Top 10 states contributing to Im[ε] (in frequency range 0.17-0.18):");
    for (rank, (idx, energy, phot_char, contrib)) in state_contributions.iter().take(10).enumerate()
    {
        println!(
            "  Rank {}: State {} | E = {:.6} | x_pp² = {:.6} | photonic = {:.4} | contrib = {:.2e}",
            rank + 1,
            idx,
            energy,
            x_pp[*idx] * x_pp[*idx],
            phot_char,
            contrib
        );
    }

    for (w_idx, &omega) in omegas.iter().enumerate() {
        let mut eps_contrib = c64::new(0.0, 0.0);
        let mut weight_sum = 0.0;
        let mut photonic_sum = 0.0;

        for i in 0..pp_vals.len() {
            let delta_e = pp_vals[i] - ground_state_energy;
            let contrib = x_pp[i] * x_pp[i] / (omega - delta_e + c64::i() * gamma);
            eps_contrib += contrib;

            // Weight by the absolute value of the imaginary part of the contribution (absorption)
            let weight = contrib.im.abs();
            weight_sum += weight;
            photonic_sum += weight * photonic_chars[i];
        }

        im_eps[w_idx] = (1.0 - 4.0 * PI * eps_contrib).im.abs();

        if weight_sum > 1e-12 {
            photonic_weighted[w_idx] = photonic_sum / weight_sum;
        }
    }

    // Set up the plot
    let resolution = if zoomed {
        (1600, 1000)
    } else {
        (1600 * 2, 1000 * 2)
    };
    let root = SVGBackend::new(save_path, resolution).into_drawing_area();
    root.fill(&WHITE)?;

    let (x_min, x_max) = omega_range.unwrap_or((omegas[0], omegas[omegas.len() - 1]));
    let y_min = 0.0;
    let mut y_max = im_eps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Include zero-coupling data in y-axis scaling if available
    if let Some(zero_im) = zero_coupling_imag {
        let zero_max = zero_im.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        y_max = y_max.max(zero_max);
    }

    let y_max = y_max * 1.05;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Im[ε(ω)] - Photonic vs Non-Photonic Contributions",
            ("sans-serif", 30),
        )
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Frequency (a.u.)")
        .y_desc("Im[ε(ω)]")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 18))
        .draw()?;

    // Draw the two stacked layers
    // Layer 1 (bottom, blue): non-photonic contribution
    let non_photonic_color = RGBColor(50, 100, 200);
    let mut non_photonic_points: Vec<(f64, f64)> = Vec::new();
    for i in 0..omegas.len() {
        let height = im_eps[i] * (1.0 - photonic_weighted[i]);
        non_photonic_points.push((omegas[i], height));
    }
    chart
        .draw_series(AreaSeries::new(
            non_photonic_points,
            0.0,
            non_photonic_color.filled(),
        ))?
        .label("Non-Photonic")
        .legend(move |(x, y)| {
            Rectangle::new([(x, y - 5), (x + 20, y + 5)], non_photonic_color.filled())
        });

    // Layer 2 (top, red): photonic contribution stacked on top
    let photonic_color = RGBColor(200, 50, 50);
    let mut photonic_stacked_points: Vec<(f64, f64)> = Vec::new();
    for i in 0..omegas.len() {
        // let base_height = im_eps[i] * (1.0 - photonic_weighted[i]);
        let total_height = im_eps[i];
        photonic_stacked_points.push((omegas[i], total_height));
    }

    // For proper stacking, we need to create a polygon that fills between the base and top
    let mut photonic_polygon: Vec<(f64, f64)> = Vec::new();
    for i in 0..omegas.len() {
        let base_height = im_eps[i] * (1.0 - photonic_weighted[i]);
        photonic_polygon.push((omegas[i], base_height));
    }
    for i in (0..omegas.len()).rev() {
        photonic_polygon.push((omegas[i], im_eps[i]));
    }

    chart
        .draw_series(std::iter::once(Polygon::new(
            photonic_polygon,
            photonic_color.filled(),
        )))?
        .label("Photonic")
        .legend(move |(x, y)| {
            Rectangle::new([(x, y - 5), (x + 20, y + 5)], photonic_color.filled())
        });

    // Draw zero-coupling dielectric function as black dashed line if provided
    if let Some(zero_im) = zero_coupling_imag {
        chart
            .draw_series(LineSeries::new(
                omegas.iter().zip(zero_im.iter()).map(|(&x, &y)| (x, y)),
                &BLACK,
            ))?
            .label("Zero Coupling")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
    }

    // Draw photon frequencies if provided
    if let Some(qs) = q_values {
        plot_photon_freqs!(chart, y_min, y_max, qs);
    }

    // Add legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.9))
        .border_style(&BLACK)
        .label_font(("sans-serif", 18))
        .draw()?;

    root.present()?;
    Ok(())
}

/// Plot the imaginary part of the dielectric function as a multi-layer stacked area
/// showing contributions from each phonon Fock state (0, 1, 2, ..., n_fock).
///
/// At each frequency ω, the plot decomposes Im[ε(ω)] into layers based on which
/// phonon Fock state each eigenstate primarily occupies.
///
/// This requires the raw eigenstate data (pp_vals, x_pp, phonon_numbers) and
/// knowledge of n_fock to bin the states.
pub fn plot_dielectric_stacked_phonon(
    omegas: &Array1<f64>,
    pp_vals: &Array1<f64>,
    x_pp: &Array1<f64>,
    phonon_numbers: &[f64],
    ground_state_index: usize,
    n_fock: usize,
    gamma: f64,
    save_path: &str,
    zoomed: bool,
    omega_range: Option<(f64, f64)>,
    q_values: Option<&Vec<Vector3D>>,
    zero_coupling_imag: Option<&Array1<f64>>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Compute the imaginary part and contributions from each Fock state
    let ground_state_energy = pp_vals[ground_state_index];
    let mut im_eps = Array1::<f64>::zeros(omegas.len());

    // For each Fock state, compute its weighted contribution at each frequency
    let mut fock_contributions: Vec<Array1<f64>> =
        vec![Array1::<f64>::zeros(omegas.len()); n_fock + 1];

    for (w_idx, &omega) in omegas.iter().enumerate() {
        let mut eps_contrib = c64::new(0.0, 0.0);
        let mut total_weight = 0.0;
        let mut fock_weights: Vec<f64> = vec![0.0; n_fock + 1];

        for i in 0..pp_vals.len() {
            if i == ground_state_index {
                continue;
            }
            let delta_e = pp_vals[i] - ground_state_energy;
            let contrib = x_pp[i] * x_pp[i] / (omega - delta_e + c64::i() * gamma);
            eps_contrib += contrib;

            // Weight by the absolute value of the imaginary part of the contribution (absorption)
            let weight = contrib.im.abs();
            total_weight += weight;

            // Assign this state to the nearest Fock state
            let fock_state = phonon_numbers[i].round() as usize;
            let fock_idx = fock_state.min(n_fock);
            fock_weights[fock_idx] += weight;
        }

        im_eps[w_idx] = (1.0 - 4.0 * PI * eps_contrib).im.abs();

        // Normalize the Fock contributions to sum to im_eps[w_idx]
        if total_weight > 1e-12 {
            for fock_idx in 0..=n_fock {
                fock_contributions[fock_idx][w_idx] =
                    im_eps[w_idx] * (fock_weights[fock_idx] / total_weight);
            }
        }
    }

    // Set up the plot
    let resolution = if zoomed {
        (1600, 1000)
    } else {
        (1600 * 2, 1000 * 2)
    };
    let root = SVGBackend::new(save_path, resolution).into_drawing_area();
    root.fill(&WHITE)?;

    let (x_min, x_max) = omega_range.unwrap_or((omegas[0], omegas[omegas.len() - 1]));
    let y_min = 0.0;
    let mut y_max = im_eps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Include zero-coupling data in y-axis scaling if available
    if let Some(zero_im) = zero_coupling_imag {
        let zero_max = zero_im.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        y_max = y_max.max(zero_max);
    }

    let y_max = y_max * 1.05;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Im[ε(ω)] - Phonon Fock State Contributions",
            ("sans-serif", 30),
        )
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Frequency (a.u.)")
        .y_desc("Im[ε(ω)]")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 18))
        .draw()?;

    // Draw stacked layers for each Fock state
    // Use a color gradient from blue (n=0) to red (n=n_fock)
    let mut cumulative_height = Array1::<f64>::zeros(omegas.len());

    for fock_idx in 0..=n_fock {
        // Color gradient: blue -> cyan -> green -> yellow -> red
        let ratio = if n_fock > 0 {
            fock_idx as f64 / n_fock as f64
        } else {
            0.0
        };
        let color = if ratio < 0.25 {
            let t = ratio / 0.25;
            RGBColor((50.0 + t * 50.0) as u8, (100.0 + t * 150.0) as u8, 200)
        } else if ratio < 0.5 {
            let t = (ratio - 0.25) / 0.25;
            RGBColor(100, (250.0 - t * 50.0) as u8, (200.0 - t * 100.0) as u8)
        } else if ratio < 0.75 {
            let t = (ratio - 0.5) / 0.25;
            RGBColor(
                (100.0 + t * 100.0) as u8,
                (200.0 - t * 150.0) as u8,
                (100.0 - t * 100.0) as u8,
            )
        } else {
            let t = (ratio - 0.75) / 0.25;
            RGBColor(200, (50.0 - t * 0.0) as u8, (0.0) as u8)
        };

        // Create polygon for this layer
        let mut layer_polygon: Vec<(f64, f64)> = Vec::new();

        // Bottom edge (cumulative height so far)
        for i in 0..omegas.len() {
            layer_polygon.push((omegas[i], cumulative_height[i]));
        }

        // Top edge (add this Fock state's contribution)
        for i in (0..omegas.len()).rev() {
            let new_height = cumulative_height[i] + fock_contributions[fock_idx][i];
            layer_polygon.push((omegas[i], new_height));
        }

        // Update cumulative height for next layer
        for i in 0..omegas.len() {
            cumulative_height[i] += fock_contributions[fock_idx][i];
        }

        chart
            .draw_series(std::iter::once(Polygon::new(layer_polygon, color.filled())))?
            .label(format!("n = {}", fock_idx))
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.filled()));
    }

    // Draw zero-coupling dielectric function as black dashed line if provided
    if let Some(zero_im) = zero_coupling_imag {
        chart
            .draw_series(LineSeries::new(
                omegas.iter().zip(zero_im.iter()).map(|(&x, &y)| (x, y)),
                &BLACK,
            ))?
            .label("Zero Coupling")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
    }

    // Draw photon frequencies if provided
    if let Some(qs) = q_values {
        plot_photon_freqs!(chart, y_min, y_max, qs);
    }

    // Add legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.9))
        .border_style(&BLACK)
        .label_font(("sans-serif", 16))
        .draw()?;

    root.present()?;
    Ok(())
}
