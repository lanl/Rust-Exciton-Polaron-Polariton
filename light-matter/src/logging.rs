#[macro_export]
macro_rules! timed_println {
    ($($arg:tt)*) => {{
        let world_size = sim_common::mpi::mpi_size_env_or_one();
        let rank = sim_common::mpi::mpi_rank_env_or_zero();

        let mpi_message = if world_size > 1 {
            format!(" Rank {} of {}: ", rank + 1, world_size)
        } else {
            String::new()
        };

        let now = chrono::Local::now();

        println!(
            "[{}]{}{}",
            now.format("%Y-%m-%d %H:%M:%S%.3f"),
            mpi_message,
            format!($($arg)*)
        );
    }};
}

#[macro_export]
macro_rules! timed_println_rank0 {
    ($($arg:tt)*) => {{
        if sim_common::mpi::is_mpi_rank0_env() {
            $crate::timed_println!($($arg)*);
        }
    }};
}
