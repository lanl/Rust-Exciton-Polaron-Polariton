use clap::{ArgAction, Parser, Subcommand, ValueEnum};
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

    /// Slurm array chunks K points; all MPI ranks in each array task solve each K collectively.
    DistributedChunks,

    /// Merge previously computed chunk files.
    GatherChunks,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum PhotonModeArg {
    Cartesian,
    Effective,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum DielectricMethodArg {
    Legacy,
    Slab,
    Sheet2d,
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

    #[arg(long, default_value = "/lustre/scratch5/mike_tay/saved_data_lm")]
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

    #[arg(long, value_enum, default_value_t = PhotonModeArg::Cartesian)]
    pub photon_modes: PhotonModeArg,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub calculate: bool,

    #[arg(long, default_value_t = false)]
    pub use_cached: bool,

    #[arg(long, default_value_t = false)]
    pub zoomed: bool,

    /// Drop photon-recoil terms from the Hamiltonian.
    #[arg(long, default_value_t = false)]
    pub long_wavelength: bool,

    #[arg(long, default_value_t = 16)]
    pub n_rel: usize,

    #[arg(long, default_value_t = 32)]
    pub n_ex_eig: usize,

    /// Number of lowest eigenstates per K point. When --energy-cutoff is
    /// given this is only used as the initial batch-size hint.
    #[arg(long, default_value_t = 512)]
    pub n_eig: usize,

    #[arg(long, default_value_t = 128)]
    pub n_k_points: usize,

    #[arg(long, default_value_t = 16)]
    pub n_bands: usize,

    #[arg(long, default_value_t = 0.003)]
    pub total_coupling: f64,

    /// Constant multiplier applied to every generated photon-mode coupling.
    #[arg(long, default_value_t = 1.0)]
    pub photon_coupling_scale: f64,

    #[arg(long, default_value_t = 0.15)]
    pub omega_0: f64,

    /// Number of grid points per photon q direction.
    /// Total photon modes = n_modes * n_modes.
    #[arg(long, default_value_t = 23)]
    pub n_modes: usize,

    #[arg(long, default_value_t = 0.001)]
    pub q_max: f64,

    /// Effective-mode maximum photon frequency. Used with --photon-modes effective.
    #[arg(long, default_value_t = 0.2)]
    pub omega_max: f64,

    /// Effective-mode frequency spacing. Used with --photon-modes effective.
    #[arg(long, default_value_t = 1e-4)]
    pub d_omega: f64,

    /// Effective-mode reflectivity. Used with --photon-modes effective.
    #[arg(long, default_value_t = 0.999999)]
    pub reflectivity: f64,

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

    /// Compute ALL eigenstates with energy <= this value at every K point
    /// (overrides --n-eig). Also used as the upper y-limit of the band plot.
    #[arg(long, allow_negative_numbers = true)]
    pub energy_cutoff: Option<f64>,

    #[command(flatten)]
    pub ckpt: CheckpointArgs,
}

#[derive(Debug, Clone, Parser)]
pub struct DielectricArgs {
    #[command(flatten)]
    pub exciton: ExcitonArgs,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub calculate: bool,

    #[arg(long, value_enum, default_value_t = PhotonModeArg::Effective)]
    pub photon_modes: PhotonModeArg,

    #[arg(long, value_enum, default_value_t = DielectricMethodArg::Slab)]
    pub dielectric_method: DielectricMethodArg,

    /// Exciton areal density n_X in bohr^-2.
    #[arg(long, default_value_t = 2.8e-5)]
    pub exciton_density: f64,

    /// Effective slab thickness in bohr.
    #[arg(long, default_value_t = 11.6)]
    pub slab_thickness_bohr: f64,

    /// Probe q in units of the photon-mode spacing d_q.
    #[arg(long, default_value_t = 2.0)]
    pub probe_q_in_dq: f64,

    /// Absolute probe q in bohr^-1; overrides --probe-q-in-dq.
    #[arg(long)]
    pub probe_q: Option<f64>,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub antiresonant: bool,

    /// Drop photon-recoil terms from the Hamiltonian.
    #[arg(long, default_value_t = false)]
    pub long_wavelength: bool,

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

    /// Cartesian mode q cutoff. Used with --photon-modes cartesian.
    #[arg(long, default_value_t = 0.001)]
    pub q_max: f64,

    /// Cartesian total coupling. Used with --photon-modes cartesian.
    #[arg(long, default_value_t = 0.003)]
    pub total_coupling: f64,

    /// Constant multiplier applied to every generated photon-mode coupling.
    #[arg(long, default_value_t = 1.0)]
    pub photon_coupling_scale: f64,

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

    /// Print PETSc event log at the end of the solve.
    #[arg(long, default_value_t = false)]
    pub petsc_log_view: bool,

    /// Print SLEPc EPS convergence monitor.
    #[arg(long, default_value_t = false)]
    pub eps_monitor: bool,

    /// Print SLEPc EPS configuration.
    #[arg(long, default_value_t = false)]
    pub eps_view: bool,

    /// Print PETSc matrix info.
    #[arg(long, default_value_t = false)]
    pub mat_view_info: bool,

    /// Compute ALL eigenstates with eigenvalue below this absolute energy.
    /// When set, --n-eig is ignored for the dielectric diagonalization.
    #[arg(long)]
    pub energy_cutoff: Option<f64>,

    #[command(flatten)]
    pub ckpt: CheckpointArgs,
}

#[derive(Debug, Clone, Parser)]
pub struct MatrixElemCalcArgs {
    #[command(subcommand)]
    pub target: MatrixElemCalcTarget,
}

#[derive(Debug, Clone, Subcommand)]
pub enum MatrixElemCalcTarget {
    /// Precompute LM matrix elements for a bandstructure calculation.
    Bandstructure(MatrixElemBandstructureArgs),

    /// Precompute LM matrix elements for a dielectric calculation.
    Dielectric(MatrixElemDielectricArgs),
}

#[derive(Debug, Clone, Parser)]
pub struct MatrixElemCommonArgs {
    /// Matrix-element cache directory.
    #[arg(long, default_value = "/lustre/scratch5/mike_tay/saved_data_lm/matrix_element_cache")]
    pub cache_dir: String,

    /// Force recomputation even if cache files exist.
    #[arg(long, default_value_t = false)]
    pub force_recompute: bool,
}

#[derive(Debug, Clone, Parser)]
pub struct MatrixElemBandstructureArgs {
    #[command(flatten)]
    pub bandstructure: BandstructureArgs,

    #[command(flatten)]
    pub common: MatrixElemCommonArgs,

    /// Precompute one K-point by global K index.
    ///
    /// Defaults to K_INDEX if set. If no chunk options are supplied, also falls
    /// back to SLURM_ARRAY_TASK_ID.
    #[arg(long)]
    pub k_index: Option<usize>,

    /// Precompute all K points in this job.
    #[arg(long, default_value_t = false)]
    pub all_k_points: bool,
}

#[derive(Debug, Clone, Parser)]
pub struct MatrixElemDielectricArgs {
    #[command(flatten)]
    pub dielectric: DielectricArgs,

    #[command(flatten)]
    pub common: MatrixElemCommonArgs,
}

#[derive(Debug, Clone, Parser)]
pub struct CheckpointArgs {
    /// Checkpoint eigenpairs after every batch of an --energy-cutoff solve.
    #[arg(long, default_value_t = false)]
    pub checkpoint: bool,

    /// Directory for diagonalization checkpoints.
    #[arg(long, default_value = "/lustre/scratch5/mike_tay/saved_data_lm/diag_checkpoints")]
    pub checkpoint_dir: PathBuf,

    /// Resume from existing checkpoints (set false to force a fresh solve).
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub checkpoint_resume: bool,

    /// Keep checkpoints after the final cache/chunk is written.
    #[arg(long, default_value_t = false)]
    pub keep_checkpoint: bool,
}

impl CheckpointArgs {
    pub fn to_checkpoint(&self) -> Option<crate::checkpoint::DiagCheckpoint> {
        self.checkpoint.then(|| {
            crate::checkpoint::DiagCheckpoint::new(
                self.checkpoint_dir.clone(),
                self.checkpoint_resume,
                self.keep_checkpoint,
            )
        })
    }
}