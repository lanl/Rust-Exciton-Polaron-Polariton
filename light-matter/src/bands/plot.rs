use ndarray::{Array1, s};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use crate::timed_println;

pub fn plot_color_band_structure(
    k_points: &Vec<Vec<f64>>,
    sym_points: &Vec<Vec<f64>>,
    sym_labels: &Vec<String>,
    eigvals_list: &Vec<Array1<f64>>,
    photonic_character_list: &Vec<Vec<f64>>,
    n_bands: usize,
    a0: f64,
    save_path: &str,
    v0: f64,
    coupling: f64,
    zoomed: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if k_points.len() < 2
        || sym_points.is_empty()
        || eigvals_list.is_empty()
        || eigvals_list[0].is_empty()
    {
        return Err("No color band data to plot".into());
    }

    if photonic_character_list.len() != eigvals_list.len() {
        return Err("photonic_character_list length does not match eigvals_list length".into());
    }

    let dark = false;

    // Create x-axis for the plot (path length)
    let x = sim_common::kpath::path_distances(k_points);

    // Define the high-symmetry points
    let points_per_segment = k_points.len() / sym_points.len();

    // Calculate positions of symmetry points on the x-axis
    let sym_positions: Vec<f64> = (0..sym_points.len())
        .map(|i| x[i * points_per_segment])
        .collect();

    // Check if we have any data to plot
    if eigvals_list.is_empty() || eigvals_list[0].is_empty() {
        return Err("No eigenvalue data to plot".into());
    }

    // Find min/max energy values
    let mut min_energy = f64::INFINITY;
    let mut max_energy = f64::NEG_INFINITY;

    // Limit the number of bands to plot based on available data
    let n_bands = n_bands.min(eigvals_list[0].len());

    // Find energy range
    for eigvals in eigvals_list {
        if eigvals.len() >= n_bands {
            let slice = eigvals.slice(s![0..n_bands]);
            min_energy = min_energy.min(slice.fold(f64::INFINITY, |a, &b| a.min(b)));
            max_energy = max_energy.max(slice.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        }
    }

    // Set max energy to 0.3 or use calculated max
    max_energy = 0.05;

    // Add padding to the energy range (5%)
    let energy_padding = (max_energy - min_energy) * 0.05;
    min_energy -= energy_padding;
    max_energy += energy_padding;

    if zoomed {
        min_energy = -0.04;
        max_energy = -0.01;
    }

    // Create data files for gnuplot
    let data_dir = Path::new("gnuplot_data");
    std::fs::create_dir_all(data_dir)?;

    // Create a data file for each band
    for band_idx in 0..n_bands {
        let filename = data_dir.join(format!("band_{}.dat", band_idx));
        let mut file = File::create(filename)?;

        // Write header
        writeln!(file, "# x energy photonic_character")?;

        // Write data points
        for (i, &x_val) in x.iter().enumerate() {
            if band_idx < eigvals_list[i].len() && band_idx < photonic_character_list[i].len() {
                let energy = eigvals_list[i][band_idx];
                let photonic = photonic_character_list[i][band_idx];

                // We'll handle transparency in the gnuplot script
                writeln!(file, "{} {} {}", x_val, energy, photonic)?;
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
            "set terminal pdfcairo enhanced font 'Arial,40' size 16,12 background rgb '#000000'"
        )?;
    } else {
        writeln!(
            script,
            "set terminal pdfcairo enhanced font 'Arial,40' size 16,12"
        )?;
    }
    writeln!(script, "set output '{}'", save_path)?;

    // Set margins to make room for the colorbar
    writeln!(script, "set lmargin 10")?;
    writeln!(script, "set rmargin 10")?;
    writeln!(script, "set tmargin 3")?;
    writeln!(script, "set bmargin 3")?;

    if dark {
        writeln!(
            script,
            "set title 'Exciton-Polariton Band Structure (a₀={:.1}, V₀={:.1}, g={:.6})' font 'Arial,16' textcolor rgb '#e0e0e0'",
            a0, v0, coupling
        )?;
        writeln!(script, "set xlabel 'Wave vector' textcolor rgb '#e0e0e0'")?;
        writeln!(script, "set ylabel 'Energy (a.u.)' textcolor rgb '#e0e0e0'")?;
    } else {
        writeln!(
            script,
            "set title 'Exciton-Polariton Band Structure (a₀={:.1}, V₀={:.1}, g={:.6})' font 'Arial,16'",
            a0, v0, coupling
        )?;
        writeln!(script, "set xlabel 'Wave vector'")?;
        writeln!(script, "set ylabel 'Energy (a.u.)'")?;
    }
    writeln!(script, "set xrange [{}:{}]", x[0], x[x.len() - 1])?;
    writeln!(script, "set yrange [{}:{}]", min_energy, max_energy)?;

    // Set up color palette - blue (excitonic) to red (photonic)
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
        // Draw vertical lines at symmetry points
        for &pos in &sym_positions {
            if dark {
                writeln!(
                    script,
                    "set arrow from {},{} to {},{} nohead lc rgb '#606060' lw 6",
                    pos, min_energy, pos, max_energy
                )?;
            } else {
                writeln!(
                    script,
                    "set arrow from {},{} to {},{} nohead lc rgb '#888888' lw 6",
                    pos, min_energy, pos, max_energy
                )?;
            }
        }
    }

    // Plot all bands with color mapping and transparency
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

    // Execute gnuplot
    let status = Command::new("gnuplot").arg(script_path).status()?;

    if !status.success() {
        return Err("Failed to execute gnuplot".into());
    }

    timed_println!("Band structure plot saved to {}", save_path);
    Ok(())
}
