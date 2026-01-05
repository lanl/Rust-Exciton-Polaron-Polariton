#[macro_export]
macro_rules! timed_println {
    ($($arg:tt)*) => {{
        use std::env;

        // Get MPI rank and size from environment variables
        let world_size: Option<usize> = env::var("PMI_SIZE")
            .or_else(|_| env::var("PMIX_SIZE"))
            .or_else(|_| env::var("I_MPI_SIZE"))
            .or_else(|_| env::var("OMPI_COMM_WORLD_SIZE"))
            .or_else(|_| env::var("SLURM_NTASKS"))
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        let rank: Option<usize> = env::var("PMI_RANK")
            .or_else(|_| env::var("PMIX_RANK"))
            .or_else(|_| env::var("I_MPI_RANK"))
            .or_else(|_| env::var("OMPI_COMM_WORLD_RANK"))
            .or_else(|_| env::var("SLURM_PROCID"))
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        let mpi_message: String;
        if rank.is_some() && world_size.is_some() {
            mpi_message = format!(" Rank {} of {}: ", rank.unwrap()+1, world_size.unwrap());
        }
        else {
            mpi_message = format!("");
        }

        let now = chrono::Local::now();
        println!("[{}]{}{}", now.format("%Y-%m-%d %H:%M:%S%.3f"), mpi_message, format!($($arg)*));
    }};
}

// pub fn init_logging() {
//     // Any initialization code can go here
//     timed_println!("Logging initialized");
// }