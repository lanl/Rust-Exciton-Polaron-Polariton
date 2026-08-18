mod bandstructure;
mod config;
mod dielectric;
mod matrix_elem;

use matrix_ops::Result;

use crate::cli::{Cli, Command};

pub fn run(cli: Cli) -> Result<()> {
    let num_threads = cli
        .rayon_threads
        .or_else(|| {
            std::env::var("OMP_NUM_THREADS")
                .ok()
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(1);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    match cli.command {
        Command::Bandstructure(args) => bandstructure::run_bandstructure_cli(args)?,
        Command::Dielectric(args) => dielectric::run_dielectric_cli(args)?,
        Command::MatrixElemCalc(args) => matrix_elem::run_matrix_elem_calc_cli(args)?,
    }

    Ok(())
}
