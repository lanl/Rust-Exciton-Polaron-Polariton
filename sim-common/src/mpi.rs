pub fn mpi_rank_env_or_zero() -> usize {
    std::env::var("PMI_RANK")
        .or_else(|_| std::env::var("PMIX_RANK"))
        .or_else(|_| std::env::var("I_MPI_RANK"))
        .or_else(|_| std::env::var("OMPI_COMM_WORLD_RANK"))
        .or_else(|_| std::env::var("SLURM_PROCID"))
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0)
}

pub fn mpi_size_env_or_one() -> usize {
    std::env::var("PMI_SIZE")
        .or_else(|_| std::env::var("PMIX_SIZE"))
        .or_else(|_| std::env::var("I_MPI_SIZE"))
        .or_else(|_| std::env::var("OMPI_COMM_WORLD_SIZE"))
        .or_else(|_| std::env::var("SLURM_NTASKS"))
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
}

#[inline]
pub fn is_mpi_rank0_env() -> bool {
    mpi_rank_env_or_zero() == 0
}

#[inline]
pub fn is_mpi_run_env() -> bool {
    mpi_size_env_or_one() > 1
}

pub fn env_usize(keys: &[&str]) -> Option<usize> {
    keys.iter()
        .find_map(|key| std::env::var(key).ok()?.parse::<usize>().ok())
}

pub fn chunk_bounds(total: usize, chunk_index: usize, n_chunks: usize) -> (usize, usize) {
    assert!(n_chunks > 0, "n_chunks must be positive");
    assert!(
        chunk_index < n_chunks,
        "chunk_index must be less than n_chunks"
    );

    let per_chunk = total.div_ceil(n_chunks);
    let start = chunk_index * per_chunk;
    let end = ((chunk_index + 1) * per_chunk).min(total);

    (start, end)
}
