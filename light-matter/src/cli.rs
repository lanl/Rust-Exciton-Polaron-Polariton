use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(author, version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Number of Rayon threads. Defaults to OMP_NUM_THREADS or 1.
    #[arg(long)]
    pub rayon_threads: Option<usize>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Bandstructure(BandstructureArgs),
    Dielectric(DielectricArgs),
    MatrixElemCalc(MatrixElemCalcArgs),
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum MethodArg {
    Sparse,
    Dense,
    Linop,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum BandParallelModeArg {
    /// One process/rank computes all K points.
    Serial,

    /// Old behavior: MPI ranks split K points independently.
    /// Unsafe with PETSc COMM_WORLD collective eigensolvers unless using COMM_SELF.
    KpointMpiLegacy,

    /// Slurm array chunks K points; all MPI ranks in each array task solve each K collectively.
    DistributedChunks,

    /// Merge previously computed chunk files.
    GatherChunks,
}

#[derive(Debug, Clone, Parser)]
pub struct ExcitonArgs {
    #[arg(long, default_value_t = 7)]
    pub n_kappa: usize,

    #[arg(long, default_value_t = 16.0)]
    pub a0: f64,

    #[arg(long, default_value_t = 0.05)]
    pub v0: f64,

    #[arg(long, default_value_t = 0.2)]
    pub m_e: f64,

    #[arg(long, default_value_t = 0.2)]
    pub m_h: f64,

    #[arg(long, default_value_t = 64)]
    pub grid_size: usize,

    #[arg(long, default_value_t = 3.0)]
    pub box_size: f64,

    #[arg(long, default_value = "saved_data_lm")]
    pub save_dir: PathBuf,

    #[arg(long, default_value_t = false)]
    pub recalculate: bool,

    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,

    #[arg(long, default_value_t = false)]
    pub verbose: bool,
}

impl ExcitonArgs {
    pub fn to_config(&self) -> exciton_2d::ExcitonConfig {
        exciton_2d::ExcitonConfig {
            n_kappa: self.n_kappa,
            a0: self.a0,
            v0: self.v0,
            k: vec![0.0, 0.0],
            m_e: self.m_e,
            m_h: self.m_h,
            verbose: self.verbose,
            grid_size: self.grid_size,
            box_size: self.box_size,
            save_dir: self.save_dir.clone(),
            recalculate: self.recalculate,
            n_jobs: 32,
            use_gpu: false,
            batch_size: self.batch_size,
        }
    }
}

#[derive(Debug, Clone, Parser)]
pub struct BandstructureArgs {
    #[command(flatten)]
    pub exciton: ExcitonArgs,

    #[arg(long, value_enum, default_value_t = MethodArg::Sparse)]
    pub method: MethodArg,

    #[arg(long, value_enum, default_value_t = BandParallelModeArg::DistributedChunks)]
    pub parallel: BandParallelModeArg,

    #[arg(long, default_value_t = true)]
    pub calculate: bool,

    #[arg(long, default_value_t = false)]
    pub use_cached: bool,

    #[arg(long, default_value_t = true)]
    pub zoomed: bool,

    #[arg(long, default_value_t = 16)]
    pub n_rel: usize,

    #[arg(long, default_value_t = 32)]
    pub n_ex_eig: usize,

    #[arg(long, default_value_t = 512)]
    pub n_eig: usize,

    #[arg(long, default_value_t = 128)]
    pub n_k_points: usize,

    #[arg(long, default_value_t = 16)]
    pub n_bands: usize,

    #[arg(long, default_value_t = 0.003)]
    pub total_coupling: f64,

    #[arg(long, default_value_t = 0.15)]
    pub omega_0: f64,

    /// Number of grid points per photon q direction.
    /// Total photon modes = n_modes * n_modes.
    #[arg(long, default_value_t = 23)]
    pub n_modes: usize,

    #[arg(long, default_value_t = 0.001)]
    pub q_max: f64,

    #[arg(long, default_value_t = 0.0)]
    pub pn_coupling: f64,

    #[arg(long, default_value_t = 0.0075)]
    pub pn_omega: f64,

    #[arg(long, default_value_t = 0.05)]
    pub pn_k_y: f64,

    #[arg(long, default_value_t = 0)]
    pub pn_n_fock: usize,

    /// K chunk index for distributed chunk mode.
    /// Defaults to K_CHUNK_INDEX or SLURM_ARRAY_TASK_ID.
    #[arg(long)]
    pub k_chunk_index: Option<usize>,

    /// Number of K chunks for distributed chunk mode.
    /// Defaults to K_N_CHUNKS or SLURM_ARRAY_TASK_COUNT.
    #[arg(long)]
    pub k_n_chunks: Option<usize>,
}

#[derive(Debug, Clone, Parser)]
pub struct DielectricArgs {
    #[command(flatten)]
    pub exciton: ExcitonArgs,

    #[arg(long, default_value_t = true)]
    pub calculate: bool,

    #[arg(long, default_value_t = false)]
    pub use_cached: bool,

    #[arg(long, default_value_t = false)]
    pub zoomed: bool,

    #[arg(long, default_value_t = 16)]
    pub n_rel: usize,

    #[arg(long, default_value_t = 16)]
    pub n_ex_eig: usize,

    #[arg(long, default_value_t = 1024)]
    pub n_eig: usize,

    #[arg(long, default_value_t = 0.18)]
    pub omega_0: f64,

    #[arg(long, default_value_t = 0.2)]
    pub omega_max: f64,

    #[arg(long, default_value_t = 1e-4)]
    pub d_omega: f64,

    #[arg(long, default_value_t = 32)]
    pub n_modes: usize,

    /// Comma-separated reflectivity values to scan.
    #[arg(long, value_delimiter = ',', default_value = "0.0,0.999999,0.9999999")]
    pub reflectivities: Vec<f64>,

    /// Broadening parameter. Defaults to d_omega if omitted.
    #[arg(long)]
    pub gamma: Option<f64>,

    #[arg(long, default_value_t = 0.0)]
    pub pn_coupling: f64,

    #[arg(long, default_value_t = 0.0025)]
    pub pn_omega: f64,

    #[arg(long, default_value_t = 0.05)]
    pub pn_k_x: f64,

    #[arg(long, default_value_t = 1)]
    pub pn_n_fock: usize,

    #[arg(long, default_value_t = 0.0)]
    pub omega_min_scan: f64,

    #[arg(long, default_value_t = 0.3)]
    pub omega_max_scan: f64,

    #[arg(long, default_value_t = 100000)]
    pub n_omega_points: usize,

    /// Which reflectivity index to calculate. Defaults to REFLECTIVITY_INDEX or SLURM_ARRAY_TASK_ID.
    #[arg(long)]
    pub reflectivity_index: Option<usize>,

    /// Optional plot-window minimum.
    #[arg(long)]
    pub plot_omega_min: Option<f64>,

    /// Optional plot-window maximum.
    #[arg(long)]
    pub plot_omega_max: Option<f64>,

    #[arg(long, default_value = "dielectric_scan/effective_modes/")]
    pub output_dir: String,
}

#[derive(Debug, Clone, Parser)]
pub struct MatrixElemCalcArgs {
    #[command(flatten)]
    pub exciton: ExcitonArgs,

    /// Number of relative-coordinate states.
    #[arg(long, default_value_t = 16)]
    pub n_rel: usize,

    /// Number of exciton eigenstates used as the LM basis.
    #[arg(long, default_value_t = 16)]
    pub n_ex_eig: usize,

    /// Central photon frequency.
    #[arg(long, default_value_t = 0.18)]
    pub omega_0: f64,

    /// Maximum photon frequency for effective-mode generation.
    #[arg(long, default_value_t = 0.2)]
    pub omega_max: f64,

    /// Effective-mode frequency spacing.
    #[arg(long, default_value_t = 1e-4)]
    pub d_omega: f64,

    /// Effective-mode target side length. Total target modes are n_modes*n_modes.
    #[arg(long, default_value_t = 32)]
    pub n_modes: usize,

    /// Comma-separated reflectivity values.
    #[arg(long, value_delimiter = ',', default_value = "0.0,0.999999,0.9999999")]
    pub reflectivities: Vec<f64>,

    /// Which reflectivity index to precompute.
    /// Defaults to REFLECTIVITY_INDEX or SLURM_ARRAY_TASK_ID, then 0.
    #[arg(long)]
    pub reflectivity_index: Option<usize>,

    /// Phonon coupling, included in PPConfig hash.
    #[arg(long, default_value_t = 0.0)]
    pub pn_coupling: f64,

    #[arg(long, default_value_t = 0.0025)]
    pub pn_omega: f64,

    #[arg(long, default_value_t = 0.05)]
    pub pn_k_x: f64,

    #[arg(long, default_value_t = 1)]
    pub pn_n_fock: usize,

    /// Matrix-element cache directory.
    #[arg(long, default_value = "saved_data_lm/matrix_element_cache")]
    pub cache_dir: String,

    /// Force recomputation even if cache files exist.
    #[arg(long, default_value_t = false)]
    pub force_recompute: bool,
}