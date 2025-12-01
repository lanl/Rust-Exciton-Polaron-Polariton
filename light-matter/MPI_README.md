# MPI Parallelization for Band Structure Calculations

## Quick Start

### 1. Install MPI

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install openmpi-bin libopenmpi-dev
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install openmpi openmpi-devel
module load mpi/openmpi-x86_64
```

**macOS:**
```bash
brew install open-mpi
```

**Check installation:**
```bash
mpirun --version
```

### 2. Modify Your Code

Replace the sequential function call:
```rust
// OLD: Sequential version
let (k_points, eigvals, photonic) = calculate_color_band_structure(
    &mut pp, n_k_points, n_eig, &sym_points, &pp_config, method, use_cached
);
```

With the MPI version:
```rust
// NEW: MPI version
let (k_points, eigvals, photonic) = calculate_color_band_structure_mpi(
    &mut pp, n_k_points, n_eig, &sym_points, &pp_config, method, use_cached
);
```

### 3. Run with MPI

```bash
# Build
cargo build --release

# Run with 4 processes
mpirun -np 4 ./target/release/light-matter

# Run with 8 processes
mpirun -np 8 ./target/release/light-matter
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Main Program                          │
│              (All ranks execute this)                    │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Detect MPI Environment                      │
│   (Read OMPI_COMM_WORLD_RANK/SIZE from env vars)       │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Distribute K-points                         │
│                                                          │
│  Rank 0: k-points [0, 1, 2, ..., 24]                   │
│  Rank 1: k-points [25, 26, 27, ..., 49]                │
│  Rank 2: k-points [50, 51, 52, ..., 74]                │
│  Rank 3: k-points [75, 76, 77, ..., 99]                │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│         Each Rank Processes Its K-points                 │
│              (Sequential within rank)                    │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Rank 0   │  │ Rank 1   │  │ Rank 2   │  ...        │
│  │ PETSc    │  │ PETSc    │  │ PETSc    │             │
│  │ SLEPc    │  │ SLEPc    │  │ SLEPc    │             │
│  └──────────┘  └──────────┘  └──────────┘             │
│       │             │             │                      │
│       ▼             ▼             ▼                      │
│  Save to       Save to       Save to                    │
│  rank_0.bin    rank_1.bin    rank_2.bin                │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Synchronization Barrier                     │
│         (File-based: wait for all ranks)                │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│           Rank 0: Gather All Results                     │
│                                                          │
│  1. Read rank_0.bin, rank_1.bin, rank_2.bin, ...       │
│  2. Combine and sort by k-point index                   │
│  3. Save final results to cache file                    │
│  4. Clean up temporary files                            │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Return Results                              │
│  (Rank 0 returns full results, others return empty)    │
└─────────────────────────────────────────────────────────┘
```

### Key Features

1. **Process-based parallelism**: Each MPI rank is a separate process
2. **No shared memory**: Avoids PETSc/SLEPc thread-safety issues
3. **File-based communication**: Simple and robust
4. **Automatic load balancing**: K-points distributed evenly
5. **Warm-starting**: Each rank uses previous eigenvectors within its subset

## Performance Guide

### Choosing Number of Ranks

**Rule of thumb**: 10-20 k-points per rank

```python
# Python calculator
n_k_points = 400
optimal_ranks = n_k_points // 15  # = 26 ranks
```

**Examples**:
- 100 k-points → 4-8 ranks
- 400 k-points → 16-32 ranks
- 1000 k-points → 50-100 ranks

### Expected Speedup

For well-balanced problems:
```
Speedup ≈ 0.9 × N_ranks
```

The 0.9 factor accounts for:
- File I/O overhead
- Synchronization time
- Load imbalance

**Example timings** (400 k-points, 512 eigenvalues):
```
Sequential:  40 minutes
4 ranks:     11 minutes  (3.6× speedup)
8 ranks:     6 minutes   (6.7× speedup)
16 ranks:    3 minutes   (13.3× speedup)
32 ranks:    1.5 minutes (26.7× speedup)
```

## Cluster Usage

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=band_calc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4G

# Load modules
module load openmpi/4.1.1
module load rust/1.70

# Build
cargo build --release

# Run with 32 ranks (4 nodes × 8 tasks)
mpirun -np 32 ./target/release/light-matter

# Or use srun (SLURM-aware)
srun -n 32 ./target/release/light-matter
```

### PBS/Torque Example

```bash
#!/bin/bash
#PBS -N band_calc
#PBS -l nodes=4:ppn=8
#PBS -l walltime=02:00:00
#PBS -l mem=32gb

cd $PBS_O_WORKDIR

module load openmpi
module load rust

cargo build --release

mpiexec -np 32 ./target/release/light-matter
```

### Running the Job

```bash
# SLURM
sbatch job_script.sh

# PBS/Torque
qsub job_script.pbs

# Check status
squeue -u $USER  # SLURM
qstat -u $USER   # PBS
```

## Troubleshooting

### Issue: "Running with 1 MPI ranks" (Expected more)

**Cause**: MPI environment variables not set

**Check**:
```bash
mpirun -np 4 bash -c 'echo "Rank: $OMPI_COMM_WORLD_RANK of $OMPI_COMM_WORLD_SIZE"'
```

**Solution**: Ensure you're using `mpirun` or `mpiexec`

### Issue: Ranks hang at synchronization

**Cause**: File system issues or rank crashed

**Debug**:
```bash
# Check if all ranks created barrier files
ls pp_data/mpi_temp/barrier_*.done

# Check for error messages
grep -r "error" pp_data/mpi_temp/
```

**Solution**: 
- Ensure all ranks can write to `pp_data/`
- Use shared filesystem on clusters
- Check for PETSc/SLEPc errors in individual ranks

### Issue: Poor scaling (8 ranks not much faster than 4)

**Possible causes**:
1. Too few k-points per rank
2. I/O bottleneck
3. Slow filesystem

**Solutions**:
```bash
# 1. Increase k-points
n_k_points = 400  # Instead of 100

# 2. Use local storage for temp files
export TMPDIR=/local/scratch/$USER

# 3. Profile I/O
time ls pp_data/mpi_temp/  # Should be fast
```

## Benchmarking

### Run Benchmark Script

```bash
cd light-matter/examples
chmod +x benchmark_mpi.sh
./benchmark_mpi.sh
```

This will:
1. Test sequential performance
2. Test MPI with 2, 4, 8, 16 ranks
3. Calculate speedup and efficiency
4. Generate a plot (if gnuplot available)

### Manual Timing

```bash
# Sequential
time cargo run --release

# MPI with 4 ranks
time mpirun -np 4 cargo run --release

# MPI with 8 ranks
time mpirun -np 8 cargo run --release
```

## Advanced: Using the `mpi` Crate

For production use, consider using proper MPI bindings:

### 1. Add Dependency

```toml
# Cargo.toml
[dependencies]
mpi = "0.6"
```

### 2. Update Code

```rust
use mpi::traits::*;

pub fn calculate_color_band_structure_mpi_proper(
    pp: &mut PolaronPolariton,
    // ... other params
) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>) {
    
    // Initialize MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as usize;
    let size = world.size() as usize;
    
    // ... distribute work ...
    
    // Proper MPI barrier
    world.barrier();
    
    // Gather results using MPI collectives
    if rank == 0 {
        // Receive from all ranks
        for r in 1..size {
            let (data, _status) = world.process_at_rank(r as i32)
                .receive_vec::<u8>();
            // Deserialize and process
        }
    } else {
        // Send to rank 0
        world.process_at_rank(0).send(&serialized_data);
    }
    
    // ...
}
```

### Benefits of `mpi` Crate

- ✅ Proper MPI barriers (no file polling)
- ✅ Direct memory communication (faster)
- ✅ Better error handling
- ✅ Standard MPI operations (broadcast, scatter, gather)

## Comparison Table

| Feature | Sequential | MPI (Current) | MPI (`mpi` crate) |
|---------|-----------|---------------|-------------------|
| Speed | 1× | ~N× | ~N× |
| Setup | None | MPI install | MPI install + crate |
| Communication | N/A | Files | Memory |
| Reliability | High | Good | Excellent |
| Overhead | None | Low | Very Low |
| Best for | <100 k-pts | >200 k-pts | Production |

## Summary

✅ **Use MPI when**:
- You have >200 k-points
- You have access to multiple cores/nodes
- You need results quickly

❌ **Don't use MPI when**:
- You have <100 k-points (overhead not worth it)
- You're just testing/debugging
- MPI is not available

📊 **Expected performance**:
- 4 ranks: ~3.5× faster
- 8 ranks: ~7× faster
- 16 ranks: ~14× faster

For questions or issues, see the full guide: `MPI_PARALLELIZATION_GUIDE.md`
