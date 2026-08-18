use log::info;
use ndarray::Array1;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

pub fn plot_n_colored_band_structure(
    k_points: &Vec<Vec<f64>>,
    eigvals_list: &Vec<Array1<f64>>,
    n_expectation_list: &Vec<Vec<f64>>,
    n_bands: usize,
    a0: f64,
    save_path: &str,
    v0: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    if k_points.len() < 2 || eigvals_list.is_empty() || eigvals_list[0].is_empty() {
        return Err("No band data to plot".into());
    }

    let dark = false;

    // Create x-axis for the plot (path length)
    let x = sim_common::kpath::path_distances(k_points);

    // Define the high-symmetry points
    let points_per_segment = k_points.len() / 3;

    // Positions of high-symmetry points
    let x_gamma1 = x[0];
    let x_k = x[points_per_segment - 1];
    let x_m = x[2 * points_per_segment - 1];
    let x_gamma2 = x[x.len() - 1];

    // Find min/max energy values
    let mut min_energy = f64::INFINITY;
    let mut max_energy = f64::NEG_INFINITY;

    // Find min/max n-expectation values
    let mut min_n = f64::INFINITY;
    let mut max_n = f64::NEG_INFINITY;

    // Limit the number of bands to plot
    let n_bands = n_bands.min(eigvals_list[0].len());

    // Find energy and n-expectation ranges
    for (i, eigvals) in eigvals_list.iter().enumerate() {
        let slice = eigvals.slice(ndarray::s![0..n_bands]);
        min_energy = min_energy.min(slice.fold(f64::INFINITY, |a, &b| a.min(b)));
        max_energy = max_energy.max(slice.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

        for j in 0..n_bands.min(n_expectation_list[i].len()) {
            min_n = min_n.min(n_expectation_list[i][j]);
            max_n = max_n.max(n_expectation_list[i][j]);
        }
    }

    // Add padding to the energy range (5%)
    let energy_padding = (max_energy - min_energy) * 0.05;

    max_energy = 0.05;

    min_energy -= energy_padding;
    max_energy += energy_padding;

    // Create data files for gnuplot
    let data_dir = Path::new("gnuplot_data");
    std::fs::create_dir_all(data_dir)?;

    // Create a data file for each band
    for band_idx in (0..n_bands).rev() {
        let filename = data_dir.join(format!("n_band_{}.dat", band_idx));
        let mut file = File::create(filename)?;

        // Write header
        writeln!(file, "# x energy n_expectation")?;

        // Write data points
        for (i, &x_val) in x.iter().enumerate() {
            if band_idx < eigvals_list[i].len() && band_idx < n_expectation_list[i].len() {
                let energy = eigvals_list[i][band_idx];
                let n_exp = n_expectation_list[i][band_idx];

                writeln!(file, "{} {} {}", x_val, energy, n_exp)?;
            }
        }
    }

    // Create gnuplot script
    let script_path = data_dir.join("plot_n_bands.gp");
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
            "set title 'Exciton Band Structure (a₀={:.1}, V₀={:.1})' font 'Arial,16'",
            a0, v0
        )?;
        writeln!(script, "set xlabel 'Wave vector'")?;
        writeln!(script, "set ylabel 'Energy (a.u.)'")?;
    } else {
        writeln!(
            script,
            "set title 'Exciton Band Structure (a₀={:.1}, V₀={:.1})' font 'Arial,16' textcolor rgb '#e0e0e0'",
            a0, v0
        )?;
        writeln!(script, "set xlabel 'Wave vector' textcolor rgb '#e0e0e0'")?;
        writeln!(script, "set ylabel 'Energy (a.u.)' textcolor rgb '#e0e0e0'")?;
    }

    writeln!(script, "set xrange [{}:{}]", x[0], x[x.len() - 1])?;
    writeln!(script, "set yrange [{}:{}]", min_energy, max_energy)?;

    // Set up color palette - blue (low n) to red (high n)
    writeln!(script, "set palette rgbformulae 22,13,-22")?;
    writeln!(
        script,
        "set colorbox vertical user origin 0.88,0.1 size 0.03,0.8"
    )?;
    writeln!(script, "set cbrange [{}:{}]", min_n, max_n)?;
    if dark {
        writeln!(
            script,
            "set cblabel 'Principal Quantum Number (n)' offset 2,0 textcolor rgb '#e0e0e0'"
        )?;
        writeln!(script, "set cbtics 1")?;
        writeln!(script, "set xtics textcolor rgb '#e0e0e0'")?;
        writeln!(script, "set ytics textcolor rgb '#e0e0e0'")?;
        writeln!(script, "set border linecolor rgb '#808080'")?;
    } else {
        writeln!(
            script,
            "set cblabel 'Principal Quantum Number (n)' offset 2,0"
        )?;
        writeln!(script, "set cbtics 1")?;
    }

    // Set up x-axis with symmetry point labels
    writeln!(
        script,
        "set xtics ('Γ' {}, 'X' {}, 'M' {}, 'Γ' {})",
        x_gamma1, x_k, x_m, x_gamma2
    )?;

    // Draw vertical lines at symmetry points
    for &pos in &[x_gamma1, x_k, x_m, x_gamma2] {
        writeln!(
            script,
            "set arrow from {},{} to {},{} nohead lc rgb '#888888' lw 6",
            pos, min_energy, pos, max_energy
        )?;
    }

    // Plot all bands with color mapping
    write!(script, "plot ")?;

    for band_idx in 0..n_bands {
        if band_idx > 0 {
            write!(script, ", ")?;
        }
        write!(
            script,
            "'{}' using 1:2:3 with lines lw 8 lc palette notitle",
            data_dir.join(format!("n_band_{}.dat", band_idx)).display()
        )?;
    }
    writeln!(script)?;

    // Execute gnuplot
    let status = Command::new("gnuplot").arg(script_path).status()?;

    if !status.success() {
        return Err("Failed to execute gnuplot".into());
    }

    info!("N-colored band structure plot saved to {}", save_path);
    Ok(())
}
