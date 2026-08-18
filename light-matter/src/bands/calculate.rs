use matrix_ops::{
    ConvertFrom, DiagonalizationOptions, Diagonalize, MatrixOps, MatrixOpsError, Sqrt,
};
use ndarray::{Array1, Array2, s};
use ndarray_linalg::c64;
use std::f64::EPSILON;
use std::fs::File;
use std::io::Write;
use std::iter::Sum;
use std::path::Path;
use std::process::Command;

use crate::timed_println;

use crate::config::PPConfig;
use crate::hamiltonian::{Boson, PPHamiltonian};

/// Calculate the photonic character of an eigenstate
/// Returns a value between 0 (purely excitonic) and 1 (purely photonic)
pub fn calculate_photonic_character<M, B>(
    pp: &PPHamiltonian<M, B>,
    eigenvector: &Array1<c64>,
) -> f64
where
    M: MatrixOps,
    B: Boson<M>,
{
    let n_ex = pp.polariton.exciton.n_ex_eig;
    let n_ph = pp.polariton.photon.dimension(); // +1 for ground state
    let n_pn = if pp.pn_coupling < EPSILON || pp.phonon.dimension() == 1 {
        1
    } else {
        pp.phonon.dimension()
    };

    // Calculate the total probability in photonic states (excluding ground state)
    let mut photonic_prob = 0.0;
    let mut total_prob = 0.0;

    for ex_idx in 0..n_ex {
        for ph_idx in 0..n_ph {
            for pn_idx in 0..n_pn {
                let idx = ex_idx * n_ph * n_pn + ph_idx * n_pn + pn_idx;
                let prob = eigenvector[idx].norm_sqr();

                if ph_idx > 0 {
                    // Skip photon ground state
                    photonic_prob += prob;
                }

                total_prob += prob;
            }
        }
    }

    // Return the fraction of probability in photonic states
    photonic_prob / total_prob
}

pub type ColorBandData = sim_common::data::ColoredBandData;

pub fn load_color_band_structure<M, B: Boson<M>>(
    pp: &PPHamiltonian<M, B>,
    n_k_points: usize,
    n_eig: usize,
    pp_config: &PPConfig,
    zoomed: bool,
) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>)
where
    M: MatrixOps,
{
    let save_dir = "pp_data".to_string();
    std::fs::create_dir_all(&save_dir).unwrap_or_else(|_| {});

    // Generate filename for the color band structure data
    let mut filename = format!(
        "{}/color_band_a0{}_V0{}_nrel{}_nbands{}_npoints{}_phcoup{}_nmodes{}_pncoup{}_npn{}_neig{}",
        save_dir,
        pp_config.e_config.a0,
        pp_config.e_config.v0,
        pp_config.n_rel,
        pp_config.n_bands,
        n_k_points,
        pp_config.lm_couplings.iter().sum::<f64>(),
        pp.polariton.photon.boson.get_modes().len(),
        pp_config.pn_coupling,
        pp_config.pn_n_fock,
        n_eig
    );

    filename += if zoomed { "_zoomed.bin" } else { ".bin" };

    timed_println!("Loading cached color band structure data from {}", filename);
    match ColorBandData::load_from_file(&filename) {
        Ok(data) => return data.to_results(),
        Err(e) => panic!("Failed to load cached data: {}, recalculating...", e),
    }
}

pub fn generate_band_k_points(
    n_k_points: usize,
    sym_points: &[Vec<f64>],
    a0: f64,
    zoomed: bool,
) -> Vec<Vec<f64>> {
    sim_common::kpath::generic_closed_path(n_k_points, sym_points, a0, zoomed)
}

pub fn color_band_filename(
    pp_config: &PPConfig,
    save_dir: &str,
    n_k_points: usize,
    n_eig: usize,
    zoomed: bool,
) -> String {
    let total_coupling = pp_config.lm_couplings.iter().sum::<f64>();

    format!(
        "{}/color_band_a0{}_V0{}_nrel{}_nbands{}_npoints{}_phcoup{:.6}_nmodes{}_pncoup{}_npn{}_neig{}{}.bin",
        save_dir,
        pp_config.e_config.a0,
        pp_config.e_config.v0,
        pp_config.n_rel,
        pp_config.n_bands,
        n_k_points,
        total_coupling,
        pp_config.q_values.len(),
        pp_config.pn_coupling,
        pp_config.pn_n_fock,
        n_eig,
        if zoomed { "_zoomed" } else { "" },
    )
}

pub fn color_band_chunk_filename(
    pp_config: &PPConfig,
    save_dir: &str,
    n_k_points: usize,
    n_eig: usize,
    zoomed: bool,
    chunk_index: usize,
    n_chunks: usize,
) -> String {
    let total_coupling = pp_config.lm_couplings.iter().sum::<f64>();

    format!(
        "{}/color_band_chunk_a0{}_V0{}_nrel{}_nbands{}_npoints{}_phcoup{:.6}_nmodes{}_pncoup{}_npn{}_neig{}_chunk{}_of{}{}.bin",
        save_dir,
        pp_config.e_config.a0,
        pp_config.e_config.v0,
        pp_config.n_rel,
        pp_config.n_bands,
        n_k_points,
        total_coupling,
        pp_config.q_values.len(),
        pp_config.pn_coupling,
        pp_config.pn_n_fock,
        n_eig,
        chunk_index,
        n_chunks,
        if zoomed { "_zoomed" } else { "" },
    )
}

pub fn calculate_color_band_structure_distributed_chunk<M, B>(
    pp: &mut PPHamiltonian<M, B>,
    n_k_points: usize,
    n_eig: usize,
    sym_points: &[Vec<f64>],
    pp_config: &PPConfig,
    use_cached: bool,
    zoomed: bool,
    chunk_index: usize,
    n_chunks: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>), MatrixOpsError>
where
    M: MatrixOps + Diagonalize,
    M::Scalar: Sqrt + Sum + ConvertFrom<f64> + ConvertFrom<c64>,
    B: Boson<M>,
{
    let rank = sim_common::mpi::mpi_rank_env_or_zero();
    let world_size = sim_common::mpi::mpi_size_env_or_one();

    let save_dir = "pp_data";
    std::fs::create_dir_all(save_dir).ok();

    let k_points = generate_band_k_points(n_k_points, sym_points, pp_config.e_config.a0, zoomed);

    let total_k_points = k_points.len();

    let (start_idx, end_idx) = sim_common::mpi::chunk_bounds(total_k_points, chunk_index, n_chunks);

    let chunk_filename = color_band_chunk_filename(
        pp_config,
        save_dir,
        n_k_points,
        n_eig,
        zoomed,
        chunk_index,
        n_chunks,
    );

    if rank == 0 {
        timed_println!(
            "Bandstructure distributed chunk {}/{}: K indices {}..{} of {} using {} MPI ranks",
            chunk_index,
            n_chunks,
            start_idx,
            end_idx,
            total_k_points,
            world_size,
        );
    }

    // Important: every rank checks the same cache path and exits consistently.
    if use_cached && Path::new(&chunk_filename).exists() {
        if rank == 0 {
            timed_println!("Loading cached band chunk from {}", chunk_filename);
            let data = ColorBandData::load_from_file(&chunk_filename).map_err(|e| {
                MatrixOpsError::Other {
                    message: format!("Failed to load cached chunk {}: {}", chunk_filename, e),
                }
            })?;
            return Ok(data.to_results());
        } else {
            return Ok((vec![], vec![], vec![]));
        }
    }

    let mut chunk_k_points = Vec::new();
    let mut chunk_eigvals = Vec::new();
    let mut chunk_photonic = Vec::new();

    let mut prev_eigvecs: Option<Array2<c64>> = None;

    for global_idx in start_idx..end_idx {
        let k = &k_points[global_idx];

        if rank == 0 {
            timed_println!(
                "Chunk {}/{}: solving K index {} / {}: {:?}",
                chunk_index,
                n_chunks,
                global_idx + 1,
                total_k_points,
                k,
            );
        }

        // All ranks in this Slurm array task build the same K Hamiltonian
        // and enter the same collective SLEPc solve.
        let h_pp = pp.build_pp_hamiltonian(Some(k.clone()))?;

        let mut options = DiagonalizationOptions::smallest_real(n_eig)
            .backend(matrix_ops::DiagonalizationBackend::Auto)
            .hermitian()
            .max_iterations(50_000)
            .tolerance(1e-10);

        options.initial_guess = prev_eigvecs.clone();

        let result = h_pp.diagonalize(options)?;
        let (eigvals, eigvecs) = result.into_arrays_hermitian();

        let n_returned = eigvals.len();

        let photonic_character: Vec<f64> = (0..n_returned)
            .map(|i| calculate_photonic_character(pp, &eigvecs.column(i).to_owned()))
            .collect();

        prev_eigvecs = Some(eigvecs);

        if rank == 0 {
            chunk_k_points.push(k.clone());
            chunk_eigvals.push(eigvals);
            chunk_photonic.push(photonic_character);
        }
    }

    if rank == 0 {
        let chunk_data = ColorBandData::from_results(
            chunk_k_points.clone(),
            chunk_eigvals.clone(),
            chunk_photonic.clone(),
        );

        chunk_data
            .save_to_file(&chunk_filename)
            .map_err(|e| MatrixOpsError::Other {
                message: format!("Failed to save band chunk {}: {}", chunk_filename, e),
            })?;

        timed_println!("Saved band chunk to {}", chunk_filename);

        Ok((chunk_k_points, chunk_eigvals, chunk_photonic))
    } else {
        Ok((vec![], vec![], vec![]))
    }
}

pub fn gather_color_band_structure_chunks(
    n_k_points: usize,
    n_eig: usize,
    pp_config: &PPConfig,
    zoomed: bool,
    n_chunks: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>), MatrixOpsError> {
    let save_dir = "pp_data";
    std::fs::create_dir_all(save_dir).ok();

    let mut all_k_points = Vec::new();
    let mut all_eigvals = Vec::new();
    let mut all_photonic = Vec::new();

    for chunk_index in 0..n_chunks {
        let chunk_filename = color_band_chunk_filename(
            pp_config,
            save_dir,
            n_k_points,
            n_eig,
            zoomed,
            chunk_index,
            n_chunks,
        );

        timed_println!("Loading band chunk {}", chunk_filename);

        let data =
            ColorBandData::load_from_file(&chunk_filename).map_err(|e| MatrixOpsError::Other {
                message: format!("Failed to load chunk file {}: {}", chunk_filename, e),
            })?;

        let (k, eig, phot) = data.to_results();

        all_k_points.extend(k);
        all_eigvals.extend(eig);
        all_photonic.extend(phot);
    }

    let final_filename = color_band_filename(pp_config, save_dir, n_k_points, n_eig, zoomed);

    let final_data = ColorBandData::from_results(
        all_k_points.clone(),
        all_eigvals.clone(),
        all_photonic.clone(),
    );

    final_data
        .save_to_file(&final_filename)
        .map_err(|e| MatrixOpsError::Other {
            message: format!(
                "Failed to save gathered bandstructure {}: {}",
                final_filename, e
            ),
        })?;

    timed_println!("Saved gathered bandstructure to {}", final_filename);

    Ok((all_k_points, all_eigvals, all_photonic))
}

/// Plot band structure with color-coded photonic character using gnuplot
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
    let dark = false;

    // Create x-axis for the plot (path length)
    let mut x = vec![0.0; k_points.len()];
    for i in 1..k_points.len() {
        let dx = k_points[i][0] - k_points[i - 1][0];
        let dy = k_points[i][1] - k_points[i - 1][1];
        x[i] = x[i - 1] + (dx * dx + dy * dy).sqrt();
    }

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
        min_energy = -0.03;
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
