// src/bin/main.rs
use clap::Parser;
use exciton_2d::{Exciton2DGridSystem, ExcitonConfig};
use log::{LevelFilter};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Lattice constant
    #[clap(long, default_value_t = 3.0)]
    a0: f64,
    
    /// Potential strength
    #[clap(long, default_value_t = 0.05)]
    v0: f64,
    
    /// Grid size
    #[clap(long, default_value_t = 128)]
    grid_size: usize,
    
    /// Box size
    #[clap(long, default_value_t = 3.0)]
    box_size: f64,
    
    /// Directory to save/load data
    #[clap(long, default_value = "saved_data")]
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
    #[clap(long, default_value_t = 9)]
    n_rel_states: usize,
    
    /// Number of bands to calculate
    #[clap(long, default_value_t = 16)]
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

fn plot_n_colored_bands(system: &Exciton2DGridSystem, args: &Args) {
    println!("Calculating n-colored band structure...");
    
    // Calculate band structure with n-expectation values
    let (k_points, eigvals_list, n_expectation_list) = system.calculate_n_colored_band_structure(
        600,
        args.n_rel_states,
        args.n_bands,
        args.recalculate_bands,
        0, // select_parity = 0 means both parities
    );
    
    // Plot the band structure
    let band_path = format!("{}/n_colored_band_structure_a0{}_v0{}.png", 
                           args.output_dir, system.a0, system.v0);
    
    system.plot_n_colored_band_structure(
        &k_points, 
        &eigvals_list, 
        &n_expectation_list, 
        args.n_bands,
        band_path.as_str()
    ).unwrap_or_else(|e| println!("Failed to plot n-colored band structure: {}", e));
}

fn scan_parameters(args: &Args) {
    // List of lattice constants and potential strengths to run
    let a0_list = vec![3.5];
    let v0_list = vec![0.1, 0.2];
    
    for &a0 in &a0_list {
        for &v0 in &v0_list {
            // Print configuration
            println!("Running exciton calculation with a0={}, v0={}, grid_size={}, box_size={}",
                     a0, v0, args.grid_size, args.box_size);
            
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
            let system = Exciton2DGridSystem::from_config(config);
            
            // Solve for the relative coordinate eigenstates
            let (rel_eigvals, rel_eigvecs, rel_pars) = system.solve_rel_hamiltonian(16);
            println!("Relative coordinate eigenvalues:");
            for i in 0..std::cmp::min(4, rel_eigvals.len()) {
                let title = format!("Relative Eigenstate n={} m={}", i, rel_pars[i]);
                let save_path = format!("{}/rel_eigenstate_n{}_m{}_a0{}_v0{}.png", 
                                        args.output_dir, i, rel_pars[i], a0, v0);
                println!("E_{} = {:.6} with parity {:.0}", i, rel_eigvals[i], rel_pars[i]);
                system.plot_rel_eigenstate(&rel_eigvecs.column(i).to_owned(), Some(&title), Some(&save_path));
            }
            
            // Diagonalize the full Hamiltonian
            let (eigvals, _, _, _, _) = system.diagonalize_full_hamiltonian(
                args.n_rel_states,
                args.n_bands,
                1, // select_parity = 0 means both parities
            );
            
            println!("\nFull Hamiltonian eigenvalues:");
            for i in 0..eigvals.len() {
                println!("E_{} = {:.6}", i, eigvals[i]);
            }
            
            let start_time = Instant::now();
            // Calculate and plot band structure
            let (k_points, eigvals_list, parities_list) = system.calculate_band_structure(
                args.n_k,
                args.n_rel_states,
                args.n_bands,
                args.recalculate_bands,
                0, // select_parity = 0 means both parities
            );
            let elapsed = start_time.elapsed();
    
            // Print the timing information
            println!("Band structure calculation completed in {:.2?}", elapsed);
            
            let band_path = format!("{}/band_structure_a0{}_v0{}.png", args.output_dir, a0, v0);
            system.plot_band_structure(&k_points, &eigvals_list, &parities_list, Some(&band_path));
        }
    }
}

fn main() {
    // Parse command-line arguments
    let args = Args::parse();
    
    // Set up logging
    env_logger::Builder::new()
        .filter_level(LevelFilter::Info)
        .init();
    
    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");

    // Create configuration
    let config = ExcitonConfig {
        n_kappa: 11,
        a0: 9.0,
        v0: 0.05,
        k: vec![0.0, 0.0],
        m_e: 0.3,
        m_h: 0.3,
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

    plot_n_colored_bands(&system, &args);

    scan_parameters(&args);

    
}