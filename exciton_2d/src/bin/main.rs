//! Main executable for the exciton 2D simulation
//!
//! This program calculates exciton properties in 2D materials with periodic potentials.
//! It can compute band structures, eigenstates, and other properties of the system.

use clap::Parser;
use exciton_2d::{Exciton2DGridSystem, ExcitonConfig};
use log::LevelFilter;
use matrix_ops::{ConvertFrom, Diagonalize, MatrixOps};
use ndarray::Array2;
use num_complex::Complex64;
use std::path::PathBuf;
use std::time::Instant;

/// Command-line arguments for the exciton 2D simulation
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Lattice constant
    #[clap(long, default_value_t = 3.0)]
    a0: f64,

    /// Potential strength
    #[clap(long, default_value_t = 0.1)]
    v0: f64,

    /// Grid size
    #[clap(long, default_value_t = 128)]
    grid_size: usize,

    /// Box size
    #[clap(long, default_value_t = 3.0)]
    box_size: f64,

    /// Directory to save/load data
    #[clap(long, default_value = "/lustre/scratch5/mike_tay/saved_data_ex")]
    save_dir: String,

    /// Force recalculation of relative coordinate solutions
    #[clap(long)]
    recalculate: bool,

    /// Force recalculation of band structure
    #[clap(long, default_value_t = true)]
    recalculate_bands: bool,

    /// Number of parallel jobs (-1 for all cores)
    #[clap(long, default_value_t = -1)]
    n_jobs: i32,

    /// Number of relative coordinate eigenstates
    #[clap(long, default_value_t = 16)]
    n_rel_states: usize,

    /// Number of bands to calculate
    #[clap(long, default_value_t = 35)]
    n_bands: usize,

    /// Directory to save output figures
    #[clap(long, default_value = "output_figures")]
    output_dir: String,

    /// Batch size for memory-intensive operations
    #[clap(long, default_value_t = 32)]
    batch_size: usize,

    /// Number of k-points for band structure
    #[clap(long, default_value_t = 60)]
    n_k: usize,

    /// Number of kappas for band structure
    #[clap(long, default_value_t = 11)]
    n_kappa: usize,
}

/// Calculates and plots band structure with n-expectation values as colors
///
/// # Arguments
/// * `system` - The exciton system
/// * `args` - Command-line arguments
fn plot_n_colored_bands<M: MatrixOps + Diagonalize>(system: &Exciton2DGridSystem<M>, args: &Args)
where
    M::Scalar: ConvertFrom<Complex64>,
{
    println!("Calculating n-colored band structure...");

    // Calculate band structure with n-expectation values
    let (k_points, eigvals_list, n_expectation_list) = system.calculate_n_colored_band_structure(
        150,
        args.n_rel_states,
        args.n_bands,
        args.recalculate_bands,
    );

    // Plot the band structure
    let band_path = format!(
        "{}/n_colored_band_structure_a0{}_v0{}.pdf",
        args.output_dir, system.a0, system.v0
    );

    system
        .plot_n_colored_band_structure(
            &k_points,
            &eigvals_list,
            &n_expectation_list,
            args.n_bands,
            band_path.as_str(),
        )
        .unwrap_or_else(|e| println!("Failed to plot n-colored band structure: {}", e));
}

/// Scans over multiple parameter values and calculates properties
///
/// # Arguments
/// * `args` - Command-line arguments
fn _scan_parameters(args: &Args) {
    // List of lattice constants and potential strengths to run
    let a0_list = vec![9.0];
    let v0_list = vec![0.1, 0.2, 0.3];

    for &a0 in &a0_list {
        for &v0 in &v0_list {
            // Print configuration
            println!(
                "Running exciton calculation with a0={}, v0={}, grid_size={}, box_size={}",
                a0, v0, args.grid_size, args.box_size
            );

            // Create configuration
            let config = ExcitonConfig {
                n_kappa: args.n_kappa,
                a0,
                v0,
                k: vec![0.0, 0.0],
                m_e: 1.0,
                m_h: 1.0,
                verbose: false,
                grid_size: args.grid_size,
                box_size: args.box_size,
                save_dir: PathBuf::from(&args.save_dir),
                recalculate: args.recalculate,
                n_jobs: args.n_jobs,
                use_gpu: false,
                batch_size: args.batch_size,
            };

            // Create the 2D exciton system
            let system: Exciton2DGridSystem<Array2<Complex64>> =
                Exciton2DGridSystem::from_config(config);

            // Solve for the relative coordinate eigenstates
            let (rel_eigvals, rel_eigvecs, rel_pars) = system.solve_rel_hamiltonian(16);
            println!("Relative coordinate eigenvalues:");
            for i in 0..std::cmp::min(4, rel_eigvals.len()) {
                let title = format!("Relative Eigenstate n={} m={}", i, rel_pars[i]);
                let save_path = format!(
                    "{}/rel_eigenstate_n{}_m{}_a0{}_v0{}.png",
                    args.output_dir, i, rel_pars[i], a0, v0
                );
                println!(
                    "E_{} = {:.6} with parity {:.0}",
                    i, rel_eigvals[i], rel_pars[i]
                );
                system.plot_rel_eigenstate(
                    &rel_eigvecs.column(i).to_owned(),
                    Some(&title),
                    Some(&save_path),
                );
            }

            // Diagonalize the full Hamiltonian
            let (eigvals, _, _, _) = system
                .diagonalize_full_hamiltonian(args.n_rel_states, args.n_bands)
                .unwrap();

            println!("\nFull Hamiltonian eigenvalues:");
            for i in 0..eigvals.len() {
                println!("E_{} = {:.6}", i, eigvals[i]);
            }

            let start_time = Instant::now();
            // Calculate and plot band structure
            let (k_points, eigvals_list) = system.calculate_band_structure(
                args.n_k,
                args.n_rel_states,
                args.n_bands,
                args.recalculate_bands,
            );
            let elapsed = start_time.elapsed();

            // Print the timing information
            println!("Band structure calculation completed in {:.2?}", elapsed);

            let band_path = format!("{}/band_structure_a0{}_v0{}.png", args.output_dir, a0, v0);
            system.plot_band_structure(&k_points, &eigvals_list, Some(&band_path));
        }
    }
}

fn gamma_point_eigenstates(system: &Exciton2DGridSystem<Array2<Complex64>>, args: &Args) {
    let (e_vals, _, _, _) = system
        .diagonalize_full_hamiltonian(args.n_rel_states, args.n_bands)
        .unwrap();

    let diffs = &e_vals - e_vals[0];

    println!("Exciton transitions at gamma point are: \n {:?}", diffs);
}

/// Main function
///
/// Parses command-line arguments, sets up the system, and runs calculations.
fn main() {
    // Parse command-line arguments
    let args = Args::parse();

    // Set up logging
    env_logger::Builder::new()
        .filter_level(LevelFilter::Info)
        .init();

    // Create output directory if it doesn't exist
    if let Err(e) = std::fs::create_dir_all(&args.output_dir) {
        eprintln!(
            "Warning: Failed to create output directory {}: {}",
            args.output_dir, e
        );
        eprintln!("Will attempt to continue, but saving outputs may fail.");
    }

    // Create configuration
    let config = ExcitonConfig {
        n_kappa: 11,
        a0: 16.0,
        v0: 0.03,
        k: vec![0.0, 0.0],
        m_e: 0.2,
        m_h: 0.2,
        verbose: false,
        grid_size: 64,
        box_size: 3.0,
        save_dir: PathBuf::from(&args.save_dir),
        recalculate: false,
        n_jobs: 32,
        use_gpu: false,
        batch_size: 32,
    };

    // Create the 2D exciton system
    let system = Exciton2DGridSystem::from_config(config);

    gamma_point_eigenstates(&system, &args);

    plot_n_colored_bands(&system, &args);

    // scan_parameters(&args);
}
