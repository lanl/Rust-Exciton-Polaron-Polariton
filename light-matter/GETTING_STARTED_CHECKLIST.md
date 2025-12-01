# Getting Started with MPI Parallelization - Checklist

## ☑️ Pre-Flight Checklist

### 1. Install MPI (Choose one)

#### Ubuntu/Debian
```bash
□ sudo apt-get update
□ sudo apt-get install openmpi-bin libopenmpi-dev
□ mpirun --version  # Verify installation
```

#### CentOS/RHEL
```bash
□ sudo yum install openmpi openmpi-devel
□ module load mpi/openmpi-x86_64
□ mpirun --version  # Verify installation
```

#### macOS
```bash
□ brew install open-mpi
□ mpirun --version  # Verify installation
```

### 2. Verify Your Code Compiles

```bash
□ cd light-matter
□ cargo build --release
□ # Should complete without errors
```

### 3. Test Sequential Version

```bash
□ cargo run --release
□ # Verify it produces correct results
□ # Note the time it takes
```

## ☑️ Basic MPI Setup

### 4. Update Your Code

Find where you call the band structure function (probably in `main.rs` or similar):

```rust
□ Locate this line:
  let (k_points, eigvals, photonic) = calculate_color_band_structure(
      &mut pp, n_k_points, n_eig, &sym_points, &pp_config, method, use_cached
  );

□ Change it to:
  let (k_points, eigvals, photonic) = calculate_color_band_structure_mpi(
      &mut pp, n_k_points, n_eig, &sym_points, &pp_config, method, use_cached
  );

□ Add import at top of file:
  use light_matter::color_band_plot::calculate_color_band_structure_mpi;

□ Rebuild:
  cargo build --release
```

### 5. Test MPI with 1 Rank

```bash
□ mpirun -np 1 ./target/release/light-matter
□ # Should work exactly like sequential
□ # Verify results match
```

### 6. Test MPI with 4 Ranks

```bash
□ mpirun -np 4 ./target/release/light-matter
□ # Should complete successfully
□ # Should be ~3-4x faster than sequential
```

## ☑️ Performance Verification

### 7. Run Benchmark

```bash
□ cd examples
□ chmod +x benchmark_mpi.sh
□ ./benchmark_mpi.sh
□ # Review results in benchmark_results.txt
```

### 8. Check Speedup

Expected results for 400 k-points:
```
□ Sequential:  ~40 minutes
□ 4 ranks:     ~11 minutes  (3.6x speedup)
□ 8 ranks:     ~6 minutes   (6.7x speedup)
□ 16 ranks:    ~3 minutes   (13.3x speedup)
```

If your speedup is significantly worse:
- □ Check you have enough k-points per rank (aim for 10-20)
- □ Verify file system is fast (not NFS)
- □ Check for other processes competing for CPU

## ☑️ Production Setup

### 9. Determine Optimal Configuration

```bash
□ Calculate: optimal_ranks = n_k_points / 15
□ Example: 400 k-points → 26-27 ranks optimal
□ Don't use more ranks than CPU cores available
```

### 10. For Local Machine

```bash
□ Check available cores: nproc
□ Use: mpirun -np [cores] ./target/release/light-matter
□ Monitor with: htop (in another terminal)
```

### 11. For Cluster (SLURM)

```bash
□ Create job script (see example below)
□ Submit: sbatch job.sh
□ Monitor: squeue -u $USER
□ Check results when complete
```

Example SLURM script:
```bash
#!/bin/bash
#SBATCH --job-name=band_calc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4G

module load openmpi
module load rust

cd $SLURM_SUBMIT_DIR
cargo build --release

mpirun -np 32 ./target/release/light-matter
```

### 12. For Cluster (PBS/Torque)

```bash
□ Create job script (see example below)
□ Submit: qsub job.pbs
□ Monitor: qstat -u $USER
□ Check results when complete
```

Example PBS script:
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

## ☑️ Verification

### 13. Verify Results

```bash
□ Check that results exist: ls pp_data/*.bin
□ Load and inspect results
□ Compare with sequential version (if available)
□ Verify plot looks correct
```

### 14. Check Performance

```bash
□ Review timing from logs
□ Calculate actual speedup
□ Verify efficiency > 80%
□ If poor, adjust number of ranks
```

### 15. Clean Up

```bash
□ Remove temporary files: rm -rf pp_data/mpi_temp
□ Archive results if needed
□ Document configuration used
```

## ☑️ Common Issues Checklist

### Problem: "Running with 1 MPI ranks" (expected more)

```bash
□ Verify you're using mpirun: mpirun -np 4 ./program
□ Check environment: echo $OMPI_COMM_WORLD_SIZE
□ Try explicit path: mpirun -np 4 /path/to/program
```

### Problem: Ranks hang at synchronization

```bash
□ Check temp directory exists: ls pp_data/mpi_temp/
□ Verify file permissions: ls -la pp_data/
□ Check all ranks wrote files: ls pp_data/mpi_temp/rank_*.bin
□ Look for errors: grep -r error pp_data/mpi_temp/
```

### Problem: Poor speedup

```bash
□ Check k-points per rank: n_k_points / n_ranks
□ Should be at least 10-20 per rank
□ Monitor CPU usage: htop
□ Check I/O is not bottleneck: iostat -x 1
□ Verify no other jobs competing for resources
```

### Problem: Results don't match sequential

```bash
□ Check convergence settings are identical
□ Verify same parameters used
□ Test with 1 rank: mpirun -np 1 ./program
□ Compare eigenvalues numerically
```

## ☑️ Documentation Reference

Quick reference to documentation:

```
□ Quick start:        MPI_README.md
□ Detailed guide:     MPI_PARALLELIZATION_GUIDE.md
□ Summary:            PARALLELIZATION_SUMMARY.md
□ This checklist:     GETTING_STARTED_CHECKLIST.md
□ Implementation:     IMPLEMENTATION_COMPLETE.md
□ Code examples:      examples/usage_example.rs
□ Run script:         examples/mpi_band_structure.sh
□ Benchmark script:   examples/benchmark_mpi.sh
```

## ☑️ First Run Quick Guide

Copy-paste these commands for your first MPI run:

```bash
# 1. Install MPI (Ubuntu)
sudo apt-get install openmpi-bin libopenmpi-dev

# 2. Build
cd light-matter
cargo build --release

# 3. Test sequential
time cargo run --release

# 4. Test MPI with 1 rank (should match sequential)
time mpirun -np 1 ./target/release/light-matter

# 5. Test MPI with 4 ranks (should be ~3-4x faster)
time mpirun -np 4 ./target/release/light-matter

# 6. Check results
ls -lh pp_data/*.bin

# 7. Benchmark
cd examples
chmod +x benchmark_mpi.sh
./benchmark_mpi.sh

# 8. Done!
```

## ☑️ Success Criteria

You know it's working when:

```
□ mpirun -np 4 completes without errors
□ Time is ~3-4x faster than sequential
□ Results match sequential version
□ All ranks show in logs
□ Speedup plot shows near-linear scaling
□ Efficiency > 80%
□ Results files created in pp_data/
□ Temporary MPI files cleaned up
```

## ☑️ Next Steps After Success

```
□ Scale up to production size (400-1000 k-points)
□ Test on cluster if available
□ Benchmark different numbers of ranks
□ Document optimal configuration for your problem
□ Set up automated runs if needed
□ Share performance results with team
```

## 📝 Notes Section

Use this space to record your configuration:

```
Date tested: _______________
System: _______________
Number of cores: _______________
K-points used: _______________
Optimal ranks found: _______________
Sequential time: _______________
MPI time (N ranks): _______________
Speedup achieved: _______________
Efficiency: _______________
Notes: 
_________________________________
_________________________________
_________________________________
```

## 🎉 Completion

When all boxes are checked, you have:

✅ MPI parallelization working  
✅ Performance verified  
✅ Production configuration determined  
✅ Ready for large-scale calculations  

**Congratulations! You're ready to use MPI parallelization!** 🚀
