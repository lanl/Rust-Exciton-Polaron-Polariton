# MPI Parallelization Guide for Band Structure Calculations

## Overview

This guide explains how to use MPI (Message Passing Interface) to parallelize the color band structure calculations. Unlike threading (Rayon), MPI is compatible with PETSc/SLEPc because each MPI rank runs in its own process with its own memory space.

## Why MPI Instead of Threading?

**PETSc/SLEPc is NOT thread-safe**, which means:
- ❌ Cannot use Rayon or any multi-threading
- ❌ Causes segmentation faults when multiple threads access PETSc
- ✅ MPI works because each rank is a separate process
- ✅ No shared memory = no race conditions

## Quick Start

### 1. Basic Usage

```rust
use light_matter::color_band_plot::calculate_color_band_structure_mpi;

// In your main function
let (k_points, eigvals, photonic) = calculate_color_band_structure_mpi(
    &mut pp,
    n_k_points,
    n_eig,
    &sym_points,
    &pp_config,
    method,
    use_cached,
);
```

### 2. Run with MPI

```bash
# Run with 4 processes
mpirun -np 4 cargo run --release

# Or on a cluster with SLURM
srun -n 8 ./target/release/light-matter

# On PBS/Torque systems
mpiexec -n 16 ./target/release/light-matter
```

## How It Works

### Work Distribution

1. **K-points are divided** across MPI ranks:
   - Total k-points: 100
   - Number of ranks: 4
   - Each rank processes: ~25 k-points

2. **Each rank processes sequentially**:
   - Rank 0: k-points 0-24
   - Rank 1: k-points 25-49
   - Rank 2: k-points 50-74
   - Rank 3: k-points 75-99

3. **Results are gathered**:
   - Each rank saves results to temporary files
   - Rank 0 collects and combines all results
   - Final data is saved to the cache file

### File-Based Synchronization

The current implementation uses a simple file-based approach:
- Each rank writes its results to `pp_data/mpi_temp/rank_N.bin`
- Barrier files (`barrier_N.done`) signal completion
- Rank 0 waits for all barrier files before gathering

## Performance Expectations

### Speedup Formula
```
Speedup ≈ N_ranks (for large problems)
```

### Example Timings
| K-points | Sequential | 4 Ranks | 8 Ranks | 16 Ranks |
|----------|-----------|---------|---------|----------|
| 100      | 10 min    | 2.5 min | 1.3 min | 0.7 min  |
| 400      | 40 min    | 10 min  | 5 min   | 2.5 min  |
| 1000     | 100 min   | 25 min  | 12.5 min| 6.3 min  |

**Note**: Actual speedup depends on:
- Problem size (larger = better scaling)
- Network speed (for distributed systems)
- Load balancing

## Advanced Usage

### 1. Optimal Number of Ranks

```bash
# Rule of thumb: Use 1 rank per ~10-20 k-points
n_k_points=200
optimal_ranks=$((n_k_points / 15))  # ~13 ranks

mpirun -np $optimal_ranks ./target/release/light-matter
```

### 2. Cluster Job Script (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=band_structure
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=02:00:00
#SBATCH --mem=64G

module load openmpi/4.1.1
module load rust/1.70

# Build in release mode
cargo build --release

# Run with MPI (32 total ranks: 2 nodes × 16 tasks)
mpirun -np 32 ./target/release/light-matter
```

### 3. PBS/Torque Job Script

```bash
#!/bin/bash
#PBS -N band_structure
#PBS -l nodes=4:ppn=8
#PBS -l walltime=02:00:00
#PBS -l mem=64gb

cd $PBS_O_WORKDIR

module load openmpi
module load rust

cargo build --release

# Run with MPI (32 total ranks: 4 nodes × 8 ppn)
mpiexec -np 32 ./target/release/light-matter
```

## Troubleshooting

### Problem: "No MPI environment detected"

**Symptom**: Program runs sequentially even with `mpirun`

**Solution**: Check environment variables:
```bash
# Should show rank and size
echo $OMPI_COMM_WORLD_RANK
echo $OMPI_COMM_WORLD_SIZE

# Or for MPICH/Intel MPI
echo $PMI_RANK
echo $PMI_SIZE
```

### Problem: "Temporary files not found"

**Symptom**: Rank 0 hangs waiting for other ranks

**Solution**: 
1. Check file permissions in `pp_data/mpi_temp/`
2. Ensure all ranks can write to the same filesystem
3. On clusters, use shared storage (not local `/tmp`)

### Problem: Poor scaling

**Symptom**: 8 ranks only 2× faster than 4 ranks

**Possible causes**:
1. **Too few k-points**: Need at least 10-20 k-points per rank
2. **I/O bottleneck**: Shared filesystem is slow
3. **Load imbalance**: Some k-points take longer than others

**Solutions**:
```rust
// Increase k-points
let n_k_points = 400;  // Instead of 100

// Use local storage for temp files (modify code)
let temp_dir = format!("/tmp/mpi_temp_{}", rank);
```

## Comparison: Sequential vs MPI

### Sequential Version
```rust
let (k_points, eigvals, photonic) = calculate_color_band_structure(
    &mut pp, n_k_points, n_eig, &sym_points, &pp_config, method, use_cached
);
```

**Pros**:
- Simple to use
- No MPI setup required
- Better for small problems (<100 k-points)
- Uses eigenvector warm-starting throughout

**Cons**:
- Slow for large problems
- Single-core only

### MPI Version
```rust
let (k_points, eigvals, photonic) = calculate_color_band_structure_mpi(
    &mut pp, n_k_points, n_eig, &sym_points, &pp_config, method, use_cached
);
```

**Pros**:
- Near-linear speedup for large problems
- Can use multiple nodes
- Compatible with PETSc/SLEPc

**Cons**:
- Requires MPI installation
- More complex setup
- Overhead for small problems
- Warm-starting only within each rank's subset

## Future Improvements

### 1. Use Proper MPI Crate

Replace environment variable detection with the `mpi` crate:

```toml
# Add to Cargo.toml
[dependencies]
mpi = "0.6"
```

```rust
use mpi::traits::*;

let universe = mpi::initialize().unwrap();
let world = universe.world();
let rank = world.rank();
let size = world.size();
```

### 2. Better Load Balancing

Distribute k-points based on estimated computation time:
```rust
// Sort k-points by expected difficulty
// Assign harder k-points to faster ranks
```

### 3. Overlap Communication and Computation

Use non-blocking MPI operations:
```rust
// Start sending results while computing next k-point
world.process_at_rank(0).immediate_send(&result);
```

## Best Practices

1. **Always test sequentially first**
   ```bash
   # Test with 1 rank
   mpirun -np 1 ./target/release/light-matter
   ```

2. **Use caching**
   ```rust
   let use_cached = true;  // Avoid recomputation
   ```

3. **Monitor resource usage**
   ```bash
   # Check CPU usage
   htop
   
   # Check MPI communication
   mpirun -np 4 --report-bindings ./target/release/light-matter
   ```

4. **Scale gradually**
   ```bash
   # Test with increasing ranks
   for n in 1 2 4 8 16; do
       echo "Testing with $n ranks"
       mpirun -np $n ./target/release/light-matter
   done
   ```

## Summary

| Feature | Sequential | MPI |
|---------|-----------|-----|
| Speed | 1× | N× (N = ranks) |
| Setup | Easy | Moderate |
| Best for | <100 k-points | >200 k-points |
| Requires | Nothing | MPI installation |
| Scaling | N/A | Near-linear |

**Recommendation**: 
- Use **sequential** for quick tests and small problems
- Use **MPI** for production runs with many k-points
