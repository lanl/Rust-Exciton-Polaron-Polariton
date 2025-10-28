use ndarray::{Array1, s};
use std::fs::File;
use std::io::{Write};
use std::path::Path;
use std::process::Command;
use crate::sing_ex::{PolaronPolariton, DiagonalizationMethod, diagonalize_pp_sing_ex};
use crate::utils::PPConfig;
use bincode::{serialize, deserialize};
use serde::{Serialize, Deserialize};
use std::error::Error;

/// Calculate the photonic character of an eigenstate
/// Returns a value between 0 (purely excitonic) and 1 (purely photonic)
pub fn calculate_photonic_character(
    pp: &PolaronPolariton,
    eigenvector: &Array1<f64>
) -> f64 {
    let n_ex = pp.exciton.n_ex_eig;
    let n_ph = pp.pp_diag_ham.photon.photon_modes.len() + 1; // +1 for ground state
    let n_pn = if pp.pp_diag_ham.phonon.is_some() {
        pp.pp_diag_ham.phonon.as_ref().unwrap().phonon_mode.n_fock
    } else {
        1
    };
    
    // Calculate the total probability in photonic states (excluding ground state)
    let mut photonic_prob = 0.0;
    let mut total_prob = 0.0;
    
    for ex_idx in 0..n_ex {
        for ph_idx in 0..n_ph {
            for pn_idx in 0..n_pn {
                let idx = ex_idx * n_ph * n_pn + ph_idx * n_pn + pn_idx;
                let prob = eigenvector[idx] * eigenvector[idx];
                
                if ph_idx > 0 {  // Skip photon ground state
                    photonic_prob += prob;
                }
                
                total_prob += prob;
            }
        }
    }

    // // Calculate the total probability in photonic states (excluding ground state)
    // let total_prob: f64 = eigenvector.iter().map(|a| a*a).sum();
    // let photonic_prob: f64 = eigenvector.to_shape((n_ex, n_ph, n_pn)).unwrap()
    //     .slice(s![..,0,..])
    //     .sum();

    // Return the fraction of probability in photonic states
    photonic_prob / total_prob
}

use rayon::prelude::*;
use std::sync::Arc;

/// Data structure for saving/loading color band structure data
#[derive(Serialize, Deserialize)]
pub struct ColorBandData {
    pub k_points: Vec<Vec<f64>>,
    pub eigenvalues: Vec<Vec<f64>>,  // Flattened Array1 to make serialization easier
    pub photonic_character: Vec<Vec<f64>>,
}

impl ColorBandData {
    /// Convert from calculation results to serializable format
    pub fn from_results(
        k_points: Vec<Vec<f64>>,
        eigenvalues: Vec<Array1<f64>>,
        photonic_character: Vec<Vec<f64>>
    ) -> Self {
        // Convert Array1 to Vec for serialization
        let eigenvalues_vec: Vec<Vec<f64>> = eigenvalues.iter()
            .map(|arr| arr.to_vec())
            .collect();
            
        Self {
            k_points,
            eigenvalues: eigenvalues_vec,
            photonic_character,
        }
    }
    
    /// Convert back to calculation result format
    pub fn to_results(&self) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>) {
        // Convert Vec back to Array1
        let eigenvalues: Vec<Array1<f64>> = self.eigenvalues.iter()
            .map(|vec| Array1::from_vec(vec.clone()))
            .collect();
            
        (self.k_points.clone(), eigenvalues, self.photonic_character.clone())
    }
    
    /// Save data to file
    pub fn save_to_file(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        File::create(filename)?;
        let data = serialize(&self)?;
        std::fs::write(filename, data)?;
        Ok(())
    }
    
    /// Load data from file
    pub fn load_from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let data = std::fs::read(filename)?;
        let band_data: Self = deserialize(&data)?;
        Ok(band_data)
    }
}

/// Calculate band structure with photonic character information
pub fn calculate_color_band_structure(
    pp: &mut PolaronPolariton,
    n_k_points: usize,
    n_eig: usize,
    sym_points: &Vec<Vec<f64>>,
    pp_config: &PPConfig,
    method: DiagonalizationMethod,
    use_cached: bool  // Flag to determine whether to load existing data
) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>) {

    let save_dir = "pp_data".to_string();
    std::fs::create_dir_all(&save_dir).unwrap_or_else(|_| {});
    
    // Generate filename for the color band structure data
    let filename = format!(
        "{}/color_band_a0{}_V0{}_nrel{}_nbands{}_npoints{}_phcoup{}_nmodes{}_pncoup{}_npn{}_neig{}.bin",
        save_dir,
        pp_config.e_config.a0,
        pp_config.e_config.v0,
        pp_config.n_rel,
        pp_config.n_bands,
        n_k_points,
        pp_config.lm_coupling,
        pp.pp_off_ham.photon.photon_modes.len(),
        pp_config.pn_coupling,
        pp_config.pn_n_fock,
        n_eig
    );
    
    // Check if cached data exists and should be used
    if use_cached && Path::new(&filename).exists() {
        println!("Loading cached color band structure data from {}", filename);
        match ColorBandData::load_from_file(&filename) {
            Ok(data) => return data.to_results(),
            Err(e) => println!("Failed to load cached data: {}, recalculating...", e)
        }
    }

    // Calculate points per segment of the path
    let points_per_seg = n_k_points / sym_points.len();

    // Generate k-points along the path
    let k_points: Vec<Vec<f64>> = sym_points.iter().enumerate().flat_map(|(sym_ind, sym_point)| {
        let sym_ind2 = (sym_ind + 1) % sym_points.len();
        (0..points_per_seg).map(move |t| {
            // Interpolate between symmetry points
            let t_frac = t as f64 / (points_per_seg - 1) as f64;
            let x = sym_point[0] + t_frac * (sym_points[sym_ind2][0] - sym_point[0]);
            let y = sym_point[1] + t_frac * (sym_points[sym_ind2][1] - sym_point[1]);

            vec![x, y]
        })
    }).collect();

    // Set up progress bar
    let total = k_points.len() as u64;
    let pb = indicatif::ProgressBar::new(total);
    pb.set_style(indicatif::ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} k-points ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    // Create a clone of the PolaronPolariton for each thread
    let pp_arc = Arc::new(pp.clone());
    
    // Calculate eigenvalues and photonic character at each k-point in parallel
    let results: Vec<(usize, Array1<f64>, Vec<f64>)> = k_points.par_iter().enumerate().map(|(idx, k)| {
        // Create a thread-local copy of the PolaronPolariton
        let mut local_pp = (*pp_arc).clone();
        local_pp.update_k(k.clone());
        
        // Diagonalize without initial guess in parallel mode
        let (eigvals, eigvecs) = diagonalize_pp_sing_ex(&local_pp, n_eig, &method, None);
        
        // Calculate photonic character for each eigenstate
        let photonic_character: Vec<f64> = (0..n_eig)
            .map(|i| calculate_photonic_character(&local_pp, &eigvecs.column(i).to_owned()))
            .collect();
        
        // Update progress bar
        pb.inc(1);
        
        // Return index with results to maintain order
        (idx, eigvals, photonic_character)
    }).collect();
    
    pb.finish_with_message("Bandstructure calculation complete");

    // Sort results by index to maintain the original order
    let mut sorted_eigvals = vec![Array1::<f64>::zeros(0); k_points.len()];
    let mut sorted_photonic = vec![Vec::<f64>::new(); k_points.len()];
    
    for (idx, eigvals, photonic) in results {
        sorted_eigvals[idx] = eigvals;
        sorted_photonic[idx] = photonic;
    }
    
    // Save the results to file
    let band_data = ColorBandData::from_results(k_points.clone(), sorted_eigvals.clone(), sorted_photonic.clone());
    match band_data.save_to_file(&filename) {
        Ok(_) => println!("Color band structure data saved to {}", filename),
        Err(e) => println!("Failed to save color band structure data: {}", e)
    }

    (k_points, sorted_eigvals, sorted_photonic)
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
    coupling: f64
) -> Result<(), Box<dyn std::error::Error>> {
    // Create x-axis for the plot (path length)
    let mut x = vec![0.0; k_points.len()];
    for i in 1..k_points.len() {
        let dx = k_points[i][0] - k_points[i-1][0];
        let dy = k_points[i][1] - k_points[i-1][1];
        x[i] = x[i-1] + (dx*dx + dy*dy).sqrt();
    }

    // Define the high-symmetry points
    let points_per_segment = k_points.len() / sym_points.len();
    
    // Calculate positions of symmetry points on the x-axis
    let sym_positions: Vec<f64> = (0..sym_points.len()).map(|i| x[i * points_per_segment]).collect();

    // Find min/max energy values
    let mut min_energy = f64::INFINITY;
    let mut max_energy = f64::NEG_INFINITY;

    // Limit the number of bands to plot
    let n_bands = n_bands.min(eigvals_list[0].len());

    // Find energy range
    for eigvals in eigvals_list {
        let slice = eigvals.slice(s![0..n_bands]);
        min_energy = min_energy.min(slice.fold(f64::INFINITY, |a, &b| a.min(b)));
        max_energy = max_energy.max(slice.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    }
    
    // Set max energy to 0.3 or use calculated max
    max_energy = max_energy.max(0.3);

    // Add padding to the energy range (5%)
    let energy_padding = (max_energy - min_energy) * 0.05;
    min_energy -= energy_padding;
    max_energy += energy_padding;

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
                writeln!(file, "{} {} {}", x_val, energy, photonic)?;
            }
        }
    }
    
    // Create gnuplot script
    let script_path = data_dir.join("plot_bands.gp");
    let mut script = File::create(&script_path)?;
    
    // Write gnuplot script
    writeln!(script, "set terminal pngcairo enhanced font 'Arial,12' size 1200,900")?;
    writeln!(script, "set output '{}'", save_path)?;
    writeln!(script, "set title 'Exciton-Polariton Band Structure (a₀={:.1}, V₀={:.1}, g={:.3})' font 'Arial,16'", a0, v0, coupling)?;
    writeln!(script, "set xlabel 'Wave vector'")?;
    writeln!(script, "set ylabel 'Energy (a.u.)'")?;
    writeln!(script, "set xrange [{}:{}]", x[0], x[x.len() - 1])?;
    writeln!(script, "set yrange [{}:{}]", min_energy, max_energy)?;
    
    // Set up color palette - blue (excitonic) to red (photonic)
    writeln!(script, "set palette defined (0 'blue', 0.5 'purple', 1 'red')")?;
    writeln!(script, "set colorbox")?;
    writeln!(script, "set cbrange [0:1]")?;
    writeln!(script, "set cblabel 'Photonic Character'")?;
    
    // Set up x-axis with symmetry point labels
    write!(script, "set xtics (")?;
    for (i, label) in sym_labels.iter().enumerate() {
        if i > 0 {
            write!(script, ", ")?;
        }
        write!(script, "'{}' {}", label, sym_positions[i])?;
    }
    writeln!(script, ")")?;
    
    // Draw vertical lines at symmetry points
    for &pos in &sym_positions {
        writeln!(script, "set arrow from {},{} to {},{} nohead lc rgb '#888888'", 
                 pos, min_energy, pos, max_energy)?;
    }
    
    // Plot all bands with color mapping
    write!(script, "plot ")?;
    for band_idx in 0..n_bands {
        if band_idx > 0 {
            write!(script, ", ")?;
        }
        write!(script, "'{}' using 1:2:3 with lines lw 2.5 palette notitle", 
               data_dir.join(format!("band_{}.dat", band_idx)).display())?;
    }
    writeln!(script)?;
    
    // Execute gnuplot
    let status = Command::new("gnuplot")
        .arg(script_path)
        .status()?;
    
    if !status.success() {
        return Err("Failed to execute gnuplot".into());
    }
    
    println!("Band structure plot saved to {}", save_path);
    Ok(())
}