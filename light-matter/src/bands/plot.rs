use ndarray::Array1;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use crate::timed_println;

const DEFAULT_PLOT_MAX_ENERGY: f64 = 0.05;
const ZOOMED_PLOT_MIN_ENERGY: f64 = -0.04;
const ZOOMED_PLOT_MAX_ENERGY: f64 = -0.01;

/// y-window of the band plot. If an energy cutoff is supplied it is used as
/// the exact upper limit (no padding), since no states exist above it.
fn band_plot_energy_window(
    data_min: f64,
    energy_cutoff: Option<f64>,
    zoomed: bool,
) -> (f64, f64) {
    let (mut min_energy, max_energy) = if zoomed {
        (
            ZOOMED_PLOT_MIN_ENERGY,
            energy_cutoff.unwrap_or(ZOOMED_PLOT_MAX_ENERGY),
        )
    } else if let Some(cutoff) = energy_cutoff {
        let padding = 0.05 * (cutoff - data_min);
        (data_min - padding, cutoff)
    } else {
        let padding = 0.05 * (DEFAULT_PLOT_MAX_ENERGY - data_min);
        (data_min - padding, DEFAULT_PLOT_MAX_ENERGY + padding)
    };

    if !(max_energy > min_energy) {
        min_energy = max_energy - 0.05 * max_energy.abs().max(1e-3);
    }

    (min_energy, max_energy)
}

pub fn plot_color_band_structure(
    k_points: &Vec<Vec<f64>>,
    sym_points: &Vec<Vec<f64>>,
    sym_labels: &Vec<String>,
    eigvals_list: &Vec<Array1<f64>>,
    photonic_character_list: &Vec<Vec<f64>>,
    n_bands: usize,
    energy_cutoff: Option<f64>,
    a0: f64,
    save_path: &str,
    v0: f64,
    coupling: f64,
    zoomed: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if k_points.len() < 2 || sym_points.is_empty() || eigvals_list.is_empty() {
        return Err("No color band data to plot".into());
    }

    if eigvals_list.len() != k_points.len() {
        return Err(format!(
            "eigvals_list has {} rows but there are {} k-points",
            eigvals_list.len(),
            k_points.len()
        )
        .into());
    }

    if photonic_character_list.len() != eigvals_list.len() {
        return Err("photonic_character_list length does not match eigvals_list length".into());
    }

    // Rows are ragged when an energy cutoff is used.
    let max_available = eigvals_list.iter().map(|e| e.len()).max().unwrap_or(0);
    if max_available == 0 {
        return Err("No eigenvalue data to plot (all K points are empty)".into());
    }

    let n_bands = match energy_cutoff {
        Some(_) => max_available,
        None => n_bands.min(max_available),
    };

    let dark = false;

    // Create x-axis for the plot (path length)
    let x = sim_common::kpath::path_distances(k_points);

    // Define the high-symmetry points
    let points_per_segment = k_points.len() / sym_points.len();

    // Calculate positions of symmetry points on the x-axis
    let sym_positions: Vec<f64> = (0..sym_points.len())
        .map(|i| x[i * points_per_segment])
        .collect();

    let data_min = eigvals_list
        .iter()
        .flat_map(|e| e.iter().take(n_bands).copied())
        .filter(|e| e.is_finite())
        .fold(f64::INFINITY, f64::min);

    if !data_min.is_finite() {
        return Err("No finite eigenvalues to plot".into());
    }

    let (min_energy, max_energy) = band_plot_energy_window(data_min, energy_cutoff, zoomed);

    timed_println!(
        "Band plot: {} bands, y-range [{:.6e}, {:.6e}] (cutoff = {:?}, zoomed = {})",
        n_bands,
        min_energy,
        max_energy,
        energy_cutoff,
        zoomed
    );

    let data_dir = Path::new("gnuplot_data");
    std::fs::create_dir_all(data_dir)?;

    // Create a data file for each band
    for band_idx in 0..n_bands {
        let filename = data_dir.join(format!("band_{}.dat", band_idx));
        let mut file = File::create(filename)?;

        // Write header
        writeln!(file, "# x energy photonic_character")?;

        // A blank line breaks the gnuplot line wherever this band index does
        // not exist (fewer states below the cutoff at that K point).
        let mut segment_open = false;

        for (i, &x_val) in x.iter().enumerate() {
            let point = eigvals_list[i]
                .get(band_idx)
                .copied()
                .zip(photonic_character_list[i].get(band_idx).copied());

            match point {
                Some((energy, photonic)) if energy.is_finite() && photonic.is_finite() => {
                    writeln!(file, "{} {} {}", x_val, energy, photonic)?;
                    segment_open = true;
                }
                _ => {
                    if segment_open {
                        writeln!(file)?;
                        segment_open = false;
                    }
                }
            }
        }
    }

    // Create gnuplot script
    let script_path = data_dir.join("plot_bands.gp");
    let mut script = File::create(&script_path)?;

    // Write gnuplot script
    if dark {
        writeln!(
            script,
            "set terminal svg enhanced font 'Arial,40' size 1600,1200 background rgb '#000000'"
        )?;
    } else {
        writeln!(
            script,
            "set terminal svg enhanced font 'Arial,40' size 1600,1200"
        )?;
    }
    writeln!(script, "set output '{}'", save_path)?;

    // Set margins to make room for the colorbar
    writeln!(script, "set lmargin 10")?;
    writeln!(script, "set rmargin 10")?;
    writeln!(script, "set tmargin 3")?;
    writeln!(script, "set bmargin 3")?;

    let cutoff_label = energy_cutoff
        .map(|c| format!(", E_{{cut}}={:.4e}", c))
        .unwrap_or_default();

    if dark {
        writeln!(
            script,
            "set title 'Exciton-Polariton Band Structure (a₀={:.1}, V₀={:.1}, g={:.6}{})' font 'Arial,16' textcolor rgb '#e0e0e0'",
            a0, v0, coupling, cutoff_label
        )?;
        writeln!(script, "set xlabel 'Wave vector' textcolor rgb '#e0e0e0'")?;
        writeln!(script, "set ylabel 'Energy (a.u.)' textcolor rgb '#e0e0e0'")?;
    } else {
        writeln!(
            script,
            "set title 'Exciton-Polariton Band Structure (a₀={:.1}, V₀={:.1}, g={:.6}{})' font 'Arial,16'",
            a0, v0, coupling, cutoff_label
        )?;
        writeln!(script, "set xlabel 'Wave vector'")?;
        writeln!(script, "set ylabel 'Energy (a.u.)'")?;
    }
    writeln!(script, "set xrange [{}:{}]", x[0], x[x.len() - 1])?;
    writeln!(script, "set yrange [{}:{}]", min_energy, max_energy)?;

    writeln!(script, "set palette rgbformulae 22,13,-22")?;
    writeln!(
        script,
        "set colorbox vertical user origin 0.88,0.1 size 0.03,0.8"
    )?;
    writeln!(script, "set cbrange [0:1]")?;
    if dark {
        writeln!(
            script,
            "set cblabel 'Photonic Character' offset 2,0 textcolor rgb '#e0e0e0'"
        )?;
        writeln!(script, "set cbtics 0.2 textcolor rgb '#e0e0e0'")?;
        writeln!(script, "set xtics textcolor rgb '#e0e0e0'")?;
        writeln!(script, "set ytics textcolor rgb '#e0e0e0'")?;
        writeln!(script, "set border linecolor rgb '#808080'")?;
    } else {
        writeln!(script, "set cblabel 'Photonic Character' offset 2,0")?;
        writeln!(script, "set cbtics 0.2")?;
    }

    // Define color functions based on photonic character with transparency support
    writeln!(
        script,
        "# Define RGB components for our color gradient - exactly matching the palette"
    )?;
    writeln!(script, "set angle degrees")?;
    writeln!(script, "r(x) = 3*x-1 < 0 ? 0: (3*x-1 > 1) ? 1 : 3*x-1")?;
    writeln!(script, "g(x) = sin(180*x)")?;
    writeln!(
        script,
        "b(x) = 1-(3*x-1) < 0 ? 0: (1-(3*x-1) > 1) ? 1 : 1-(3*x-1)"
    )?;

    writeln!(
        script,
        "# Define transparency function based on photonic character"
    )?;
    writeln!(
        script,
        "alpha(x) = x > 0.9 ? int(((x-0.9)/0.1*0.95)*255) : 0"
    )?;

    writeln!(script, "# Create full ARGB color")?;
    writeln!(
        script,
        "myColor(x) = (alpha(x)<<24) + (int(r(x)*0xff)<<16) + (int(g(x)*0xff)<<8) + int(b(x)*0xff)"
    )?;

    // Create a dummy file for the colorbar
    writeln!(script, "# Create a dummy file for the colorbar")?;
    let colorbar_file = data_dir.join("colorbar.dat");
    let mut cb_file = File::create(&colorbar_file)?;
    for i in 0..101 {
        let val = i as f64 / 100.0;
        writeln!(cb_file, "{} 0 {}", val, val)?;
    }

    // Set up x-axis with symmetry point labels
    write!(script, "set xtics (")?;
    for (i, label) in sym_labels.iter().enumerate() {
        if i > 0 {
            write!(script, ", ")?;
        }
        write!(script, "'{}' {}", label, sym_positions[i])?;
    }
    writeln!(script, ")")?;

    if !zoomed {
        for &pos in &sym_positions {
            let color = if dark { "#606060" } else { "#888888" };
            writeln!(
                script,
                "set arrow from {},{} to {},{} nohead lc rgb '{}' lw 6",
                pos, min_energy, pos, max_energy, color
            )?;
        }
    }

    write!(script, "plot ")?;

    // First add a dummy plot for the colorbar using palette
    write!(
        script,
        "'{}' using 1:2:3 with points pt 7 ps 0 lc palette notitle",
        colorbar_file.display()
    )?;

    // Then add all the bands
    let lw = if !zoomed { 8 } else { 16 };
    for band_idx in 0..n_bands {
        write!(script, ", ")?;
        // Use the myColor function that handles both color and transparency
        write!(
            script,
            "'{}' using 1:2:(myColor($3)) with lines lw {} lc rgb var notitle",
            data_dir.join(format!("band_{}.dat", band_idx)).display(),
            lw
        )?;
    }
    writeln!(script)?;

    // Make sure the script is closed before gnuplot reads it.
    drop(script);

    let status = Command::new("gnuplot").arg(&script_path).status()?;

    if !status.success() {
        return Err("Failed to execute gnuplot".into());
    }

    timed_println!("Band structure plot saved to {}", save_path);
    Ok(())
}