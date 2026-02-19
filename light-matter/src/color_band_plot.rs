use ndarray::{Array1, Array2, s};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{Write};
use std::path::Path;
use std::process::Command;
use crate::sing_ex::{PolaronPolariton, diagonalize_pp_sing_ex};
use crate::utils::PPConfig;
use bincode::{serialize, deserialize};
use serde::{Serialize, Deserialize};
use std::error::Error;
use std::time::Instant;
use colored::Colorize;
use ndarray_linalg::c64;

use crate::timed_println;

/// Calculate the photonic character of an eigenstate
/// Returns a value between 0 (purely excitonic) and 1 (purely photonic)
pub fn calculate_photonic_character(
    pp: &PolaronPolariton,
    eigenvector: &Array1<c64>
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
                let prob = eigenvector[idx].norm_sqr();
                
                if ph_idx > 0 {  // Skip photon ground state
                    photonic_prob += prob;
                }
                
                total_prob += prob;
            }
        }
    }

    // Return the fraction of probability in photonic states
    photonic_prob / total_prob
}

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
/// 
/// **IMPORTANT**: This function CANNOT be parallelized using Rayon or any multi-threading approach
/// because PETSc/SLEPc is NOT thread-safe. Attempting to call PETSc/SLEPc functions from multiple
/// threads simultaneously will cause segmentation faults and crashes.
/// 
/// The sequential implementation with eigenvector warm-starting is the only safe and recommended approach.
pub fn calculate_color_band_structure(
    pp: &mut PolaronPolariton,
    n_k_points: usize,
    n_eig: usize,
    sym_points: &Vec<Vec<f64>>,
    pp_config: &PPConfig,
    use_cached: bool,  // Flag to determine whether to load existing data
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
        timed_println!("Loading cached color band structure data from {}", filename);
        match ColorBandData::load_from_file(&filename) {
            Ok(data) => return data.to_results(),
            Err(e) => timed_println!("Failed to load cached data: {}, recalculating...", e)
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
    
    // Calculate eigenvalues and photonic character at each k-point sequentially
    // to use previous eigenvectors as initial guesses for better convergence
    let mut results: Vec<(usize, Array1<f64>, Vec<f64>)> = Vec::with_capacity(k_points.len());
    let mut prev_eigvecs: Option<Array2<c64>> = None;
    
    for (idx, k) in k_points.iter().enumerate() {
        // Create a thread-local copy of the PolaronPolariton
        pp.update_k(k.clone());

        let (eigvals, eigvecs) = diagonalize_pp_sing_ex(&pp, n_eig, prev_eigvecs.as_ref());
        
        // Calculate photonic character for each eigenstate
        let photonic_character: Vec<f64> = (0..n_eig)
            .map(|i| calculate_photonic_character(&pp, &eigvecs.column(i).to_owned()))
            .collect();
        
        // Store eigenvectors for next iteration
        prev_eigvecs = Some(eigvecs);
        
        // Update progress bar
        pb.inc(1);
        
        // Return index with results to maintain order
        results.push((idx, eigvals, photonic_character));
    }
    
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
        Ok(_) => timed_println!("Color band structure data saved to {}", filename),
        Err(e) => timed_println!("Failed to save color band structure data: {}", e)
    }

    (k_points, sorted_eigvals, sorted_photonic)
}

pub fn load_color_band_structure(
    pp: &mut PolaronPolariton,
    n_k_points: usize,
    n_eig: usize,
    pp_config: &PPConfig,
    zoomed: bool
) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>) {
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
        pp_config.lm_coupling,
        pp.pp_off_ham.photon.photon_modes.len(),
        pp_config.pn_coupling,
        pp_config.pn_n_fock,
        n_eig
    );

    filename += if zoomed { "_zoomed.bin" } else { ".bin" };
    
    timed_println!("Loading cached color band structure data from {}", filename);
    match ColorBandData::load_from_file(&filename) {
        Ok(data) => return data.to_results(),
        Err(e) => panic!("Failed to load cached data: {}, recalculating...", e)
    }
}

/// Calculate band structure with MPI parallelization (alternative implementation)
/// 
/// This version uses a simpler approach that's compatible with systems where
/// MPI might already be initialized by other libraries (like PETSc/SLEPc).
/// 
/// It falls back to environment variables if the mpi crate has issues.
pub fn calculate_color_band_structure_mpi_simple(
    pp: &mut PolaronPolariton,
    n_k_points: usize,
    n_eig: usize,
    sym_points: &Vec<Vec<f64>>,
    pp_config: &PPConfig,
    use_cached: bool,
    zoomed: bool
) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>) {
    use std::env;

    // Debug: Print all MPI-related environment variables
    timed_println!("\n=== MPI Environment Variables ===");
    let mpi_vars = vec![
        "PMI_SIZE", "PMI_RANK", "PMIX_SIZE", "PMIX_RANK",
        "I_MPI_SIZE", "I_MPI_RANK", "I_MPI_WORLD_SIZE", "I_MPI_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK",
        "SLURM_NTASKS", "SLURM_PROCID", "SLURM_NPROCS",
        "MPI_LOCALNRANKS", "MPI_LOCALRANKID"
    ];
    for var in &mpi_vars {
        match env::var(var) {
            Ok(val) => timed_println!("  {} = {}", var, val),
            Err(_) => timed_println!("  {} = <not set>", var),
        }
    }
    timed_println!("=================================\n");

    // Get MPI rank and size from environment variables
    // Try multiple environment variable names for different MPI implementations
    let world_size: usize = env::var("PMI_SIZE")
        .or_else(|_| env::var("PMIX_SIZE"))
        .or_else(|_| env::var("I_MPI_SIZE"))
        .or_else(|_| env::var("OMPI_COMM_WORLD_SIZE"))
        .or_else(|_| env::var("SLURM_NTASKS"))
        .expect("Could not find MPI world size in environment variables. Tried: PMI_SIZE, PMIX_SIZE, I_MPI_SIZE, OMPI_COMM_WORLD_SIZE, SLURM_NTASKS")
        .parse()
        .expect("Failed to parse world size as integer");
    
    let rank: usize = env::var("PMI_RANK")
        .or_else(|_| env::var("PMIX_RANK"))
        .or_else(|_| env::var("I_MPI_RANK"))
        .or_else(|_| env::var("OMPI_COMM_WORLD_RANK"))
        .or_else(|_| env::var("SLURM_PROCID"))
        .expect("Could not find MPI rank in environment variables. Tried: PMI_RANK, PMIX_RANK, I_MPI_RANK, OMPI_COMM_WORLD_RANK, SLURM_PROCID")
        .parse()
        .expect("Failed to parse rank as integer");

    if rank == 0 {
        timed_println!("Running with {} MPI ranks (simple mode)", world_size);
    }

    // Start timer on rank 0
    let start_time = if rank == 0 {
        Some(Instant::now())
    } else {
        None
    };

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
        pp_config.lm_coupling,
        pp.pp_off_ham.photon.photon_modes.len(),
        pp_config.pn_coupling,
        pp_config.pn_n_fock,
        n_eig
    );

    filename += if zoomed { "_zoomed.bin" } else { ".bin" };
    
    // Check if cached data exists and should be used (only rank 0)
    if rank == 0 && use_cached && Path::new(&filename).exists() {
        timed_println!("Loading cached color band structure data from {}", filename);
        match ColorBandData::load_from_file(&filename) {
            Ok(data) => return data.to_results(),
            Err(e) => timed_println!("Failed to load cached data: {}, recalculating...", e)
        }
    }

    // Calculate points per segment of the path
    let points_per_seg = n_k_points / sym_points.len();

    // Generate k-points along the path
    let k_points: Vec<Vec<f64>>;
    if zoomed {
        let start = vec![3. / 4. * PI / pp_config.e_config.a0, 0.];
        let end =   vec![PI / pp_config.e_config.a0, 0.];

        timed_println!("Performing zoomed in band structure from k = {:?} to {:?}", start, end);
        k_points = (0 .. n_k_points).map(|k_ind|{
            let t_frac = k_ind as f64 / (n_k_points - 1) as f64;

            vec![start[0] + t_frac * (end[0] - start[0]), start[1] + t_frac * (end[1] - start[1])]
        }).collect()
    }
    else {
        k_points = sym_points.iter().enumerate().flat_map(|(sym_ind, sym_point)| {
            let sym_ind2 = (sym_ind + 1) % sym_points.len();
            (0..points_per_seg).map(move |t| {
                // Interpolate between symmetry points
                let t_frac = t as f64 / (points_per_seg - 1) as f64;
                let x = sym_point[0] + t_frac * (sym_points[sym_ind2][0] - sym_point[0]);
                let y = sym_point[1] + t_frac * (sym_points[sym_ind2][1] - sym_point[1]);

                vec![x, y]
            })
        }).collect();
    }

    // Distribute k-points across ranks
    let total_k_points = k_points.len();
    let k_per_rank = (total_k_points + world_size - 1) / world_size;
    let start_idx = rank * k_per_rank;
    let end_idx = ((rank + 1) * k_per_rank).min(total_k_points);
    
    let my_k_points: Vec<(usize, Vec<f64>)> = k_points[start_idx..end_idx]
        .iter()
        .enumerate()
        .map(|(local_idx, k)| (start_idx + local_idx, k.clone()))
        .collect();

    if rank == 0 {
        timed_println!("Processing {} k-points across {} ranks", total_k_points, world_size);
    }
    timed_println!("Rank {}: Processing k-points {} to {}", rank, start_idx, end_idx - 1);

    
    // Calculate eigenvalues and photonic character for this rank's k-points
    let mut local_results: Vec<(usize, Array1<f64>, Vec<f64>)> = Vec::with_capacity(my_k_points.len());
    // let mut prev_eigvecs: Option<Array2<c64>> = None;

    for (global_idx, k) in my_k_points.iter() {
        pp.update_k(k.clone());

        // // Use warm start with only a small subset of previous eigenvectors
        // // IMPORTANT: Only pass 5 initial vectors to avoid over-constraining the solver
        // // Passing too many vectors can lead to noisy results
        // let initial_guess_subset = prev_eigvecs.as_ref().map(|vecs| {
        //     let n_init = 5.min(vecs.ncols());
        //     vecs.slice(s![.., 0..n_init]).to_owned()
        // });
        // let (eigvals, eigvecs) = diagonalize_pp_sing_ex(&pp, n_eig, initial_guess_subset.as_ref()); 
        let (eigvals, eigvecs) = diagonalize_pp_sing_ex(&pp, n_eig, None); 

        
        timed_println!("Calculating photonic character");
        let photonic_character: Vec<f64> = (0..n_eig)
            .map(|i| calculate_photonic_character(&pp, &eigvecs.column(i).to_owned()))
            .collect();
        
        // prev_eigvecs = Some(eigvecs);
        local_results.push((*global_idx, eigvals, photonic_character));

        // Calculate ETA on rank 0
        if let Some(start) = start_time {
            let elapsed_secs = start.elapsed().as_secs_f64();
            let completed_k_points = local_results.len();
            if completed_k_points > 0 {
                let avg_time_per_k_point = elapsed_secs / completed_k_points as f64;
                let remaining_k_points_this_rank = my_k_points.len() - completed_k_points;
                let eta = avg_time_per_k_point * remaining_k_points_this_rank as f64;

                if rank == 0 {
                    let eta_str = format_time(eta);
                    let elapsed_str = format_time(elapsed_secs);

                    timed_println!(
                        "\n\n{}\n⏱️ ETA: {} | Rank: {} | K-point: {}/{} | Elapsed: {}\n\n",
                        "ETA UPDATE".red(),
                        eta_str.red(),
                        rank,
                        completed_k_points,
                        my_k_points.len(),
                        elapsed_str.red()
                    );
                }
            }
        }
    }

    timed_println!("Rank {}: Completed {} k-points", rank, local_results.len());

    // Save local results to temporary files (file-based communication)
    // Make temp_dir unique per job using SLURM_JOB_ID or process ID
    let job_id = env::var("SLURM_JOB_ID")
        .or_else(|_| env::var("SLURM_JOBID"))
        .unwrap_or_else(|_| std::process::id().to_string());
    let temp_dir = format!("{}/mpi_temp_{}", save_dir, job_id);
    std::fs::create_dir_all(&temp_dir).unwrap_or_else(|_| {});
    let temp_filename = format!("{}/rank_{}.bin", temp_dir, rank);
    
    let local_data = ColorBandData::from_results(
        local_results.iter().map(|(idx, _, _)| k_points[*idx].clone()).collect(),
        local_results.iter().map(|(_, eigvals, _)| eigvals.clone()).collect(),
        local_results.iter().map(|(_, _, photonic)| photonic.clone()).collect(),
    );
    
    if let Err(e) = local_data.save_to_file(&temp_filename) {
        timed_println!("Rank {}: Failed to save temporary data: {}", rank, e);
    }

    // Improved barrier synchronization with unique job-specific directory
    let barrier_dir = format!("{}/barriers", temp_dir);
    std::fs::create_dir_all(&barrier_dir).unwrap();
    let barrier_file = format!("{}/rank_{}.done", barrier_dir, rank);
    
    // Write rank completion marker with timestamp to avoid stale files
    std::fs::write(&barrier_file, format!("{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs())).unwrap();
    
    if rank == 0 {
        timed_println!("Rank 0: Waiting for all ranks to finish...");
        let mut last_count = 0;
        loop {
            let mut completed_count = 0;
            for r in 0..world_size {
                let barrier_check = format!("{}/rank_{}.done", barrier_dir, r);
                if Path::new(&barrier_check).exists() {
                    completed_count += 1;
                }
            }
            if completed_count != last_count {
                timed_println!("Rank 0: {} of {} ranks completed", completed_count, world_size);
                last_count = completed_count;
            }
            if completed_count == world_size {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        timed_println!("Rank 0: All ranks finished, gathering results...");
    }
    
    // Gather results on rank 0
    if rank == 0 {
        let mut all_results: Vec<(usize, Array1<f64>, Vec<f64>)> = Vec::with_capacity(n_k_points);

        for r in 0..world_size {
            let rank_file = format!("{}/rank_{}.bin", temp_dir, r);
            if let Ok(rank_data) = ColorBandData::load_from_file(&rank_file) {
                let (_, eigvals, photonic) = rank_data.to_results();
                let start = r * k_per_rank;
                let _end = ((r + 1) * k_per_rank).min(n_k_points);

                for (local_idx, (eigval, phot)) in eigvals.iter().zip(photonic.iter()).enumerate() {
                    let global_idx = start + local_idx;
                    if global_idx < n_k_points {
                        all_results.push((global_idx, eigval.clone(), phot.clone()));
                    }
                }
            }

            // Progress update (no ETA during gathering)
            if r % 1 == 0 {
                timed_println!("Rank 0: Gathered results from rank {}/{}", r + 1, world_size);
            }
        }

        all_results.sort_by_key(|(idx, _, _)| *idx);
        
        let mut sorted_eigvals = vec![Array1::<f64>::zeros(0); total_k_points];
        let mut sorted_photonic = vec![Vec::<f64>::new(); total_k_points];
        
        for (idx, eigvals, photonic) in all_results {
            sorted_eigvals[idx] = eigvals;
            sorted_photonic[idx] = photonic;
        }
        
        let band_data = ColorBandData::from_results(k_points.clone(), sorted_eigvals.clone(), sorted_photonic.clone());
        match band_data.save_to_file(&filename) {
            Ok(_) => timed_println!("Color band structure data saved to {}", filename),
            Err(e) => timed_println!("Failed to save color band structure data: {}", e)
        }
        
        let _ = std::fs::remove_dir_all(&temp_dir);
        
        (k_points, sorted_eigvals, sorted_photonic)
    } else {
        (vec![], vec![], vec![])
    }
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
    zoomed: bool
) -> Result<(), Box<dyn std::error::Error>> {

    let dark = true;

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
    max_energy = 0.0;

    // Add padding to the energy range (5%)
    let energy_padding = (max_energy - min_energy) * 0.05;
    min_energy -= energy_padding;
    max_energy += energy_padding;

    if zoomed {
        min_energy = -0.195;
        max_energy = -0.16;
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
        writeln!(script, "set terminal pdfcairo enhanced font 'Arial,40' size 16,12 background rgb '#000000'")?;
    }
    else {
        writeln!(script, "set terminal pdfcairo enhanced font 'Arial,40' size 16,12")?;
    }
    writeln!(script, "set output '{}'", save_path)?;
    
    // Set margins to make room for the colorbar
    writeln!(script, "set lmargin 10")?;
    writeln!(script, "set rmargin 10")?;
    writeln!(script, "set tmargin 3")?;
    writeln!(script, "set bmargin 3")?;
        
    if dark {
        writeln!(script, "set title 'Exciton-Polariton Band Structure (a₀={:.1}, V₀={:.1}, g={:.3})' font 'Arial,16' textcolor rgb '#e0e0e0'", a0, v0, coupling)?;
        writeln!(script, "set xlabel 'Wave vector' textcolor rgb '#e0e0e0'")?;
        writeln!(script, "set ylabel 'Energy (a.u.)' textcolor rgb '#e0e0e0'")?;
    }
    else {
        writeln!(script, "set title 'Exciton-Polariton Band Structure (a₀={:.1}, V₀={:.1}, g={:.3})' font 'Arial,16'", a0, v0, coupling)?;
        writeln!(script, "set xlabel 'Wave vector'")?;
        writeln!(script, "set ylabel 'Energy (a.u.)'")?;
    }
    writeln!(script, "set xrange [{}:{}]", x[0], x[x.len() - 1])?;
    writeln!(script, "set yrange [{}:{}]", min_energy, max_energy)?;
    
    // Set up color palette - blue (excitonic) to red (photonic)
    writeln!(script, "set palette rgbformulae 22,13,-22")?;
    writeln!(script, "set colorbox vertical user origin 0.88,0.1 size 0.03,0.8")?;
    writeln!(script, "set cbrange [0:1]")?;
    if dark {
        writeln!(script, "set cblabel 'Photonic Character' offset 2,0 textcolor rgb '#e0e0e0'")?;
        writeln!(script, "set cbtics 0.2 textcolor rgb '#e0e0e0'")?;
        writeln!(script, "set xtics textcolor rgb '#e0e0e0'")?;
        writeln!(script, "set ytics textcolor rgb '#e0e0e0'")?;
        writeln!(script, "set border linecolor rgb '#808080'")?;
    }
    else {
        writeln!(script, "set cblabel 'Photonic Character' offset 2,0")?;
        writeln!(script, "set cbtics 0.2")?;
    }
    
    // Define color functions based on photonic character with transparency support
    writeln!(script, "# Define RGB components for our color gradient - exactly matching the palette")?;
    writeln!(script, "set angle degrees")?;
    writeln!(script, "r(x) = 3*x-1 < 0 ? 0: (3*x-1 > 1) ? 1 : 3*x-1")?;
    writeln!(script, "g(x) = sin(180*x)")?;
    writeln!(script, "b(x) = 1-(3*x-1) < 0 ? 0: (1-(3*x-1) > 1) ? 1 : 1-(3*x-1)")?;
    
    writeln!(script, "# Define transparency function based on photonic character")?;
    writeln!(script, "alpha(x) = x > 0.9 ? int(((x-0.9)/0.1*0.95)*255) : 0")?;
    
    writeln!(script, "# Create full ARGB color")?;
    writeln!(script, "myColor(x) = (alpha(x)<<24) + (int(r(x)*0xff)<<16) + (int(g(x)*0xff)<<8) + int(b(x)*0xff)")?;
    
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
                writeln!(script, "set arrow from {},{} to {},{} nohead lc rgb '#606060' lw 6", 
                        pos, min_energy, pos, max_energy)?;
            }
            else {
                writeln!(script, "set arrow from {},{} to {},{} nohead lc rgb '#888888' lw 6", 
                        pos, min_energy, pos, max_energy)?;
            }
        }
    }
    
    // Plot all bands with color mapping and transparency
    write!(script, "plot ")?;
    
    // First add a dummy plot for the colorbar using palette
    write!(script, "'{}' using 1:2:3 with points pt 7 ps 0 lc palette notitle", 
           colorbar_file.display())?;
    
    // Then add all the bands
    let lw = if !zoomed {8} else {16};
    for band_idx in 0..n_bands {
        write!(script, ", ")?;
        // Use the myColor function that handles both color and transparency
        write!(script, "'{}' using 1:2:(myColor($3)) with lines lw {} lc rgb var notitle", 
               data_dir.join(format!("band_{}.dat", band_idx)).display(), lw)?;
    }
    writeln!(script)?;
    
    // Execute gnuplot
    let status = Command::new("gnuplot")
        .arg(script_path)
        .status()?;
    
    if !status.success() {
        return Err("Failed to execute gnuplot".into());
    }
    
    timed_println!("Band structure plot saved to {}", save_path);
    Ok(())
}

fn format_time(secs: f64) -> String {
    let hours = (secs / 3600.0).floor() as u64;
    let minutes = ((secs % 3600.0) / 60.0).floor() as u64;
    let seconds = (secs % 60.0).round() as u64;

    format!("{} hrs {} min {:.2} sec", hours, minutes, seconds)
}