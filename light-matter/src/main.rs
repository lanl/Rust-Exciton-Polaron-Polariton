use clap::Parser;
use light_matter::{app, cli::Cli};

fn main() -> matrix_ops::Result<()> {
    let cli = Cli::parse();

    matrix_ops::with_slepc(|| app::run(cli))
}
