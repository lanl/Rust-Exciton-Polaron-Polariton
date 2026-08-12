#!/bin/bash
#
#SBATCH --job-name=sing_ex_disp
#SBATCH --output myjob_%j.out
#SBATCH --error myjob_%j.err
#SBATCH --partition=skylake-gold
#
#SBATCH --time=2-00:00:00
###SBATCH --time=8:00:00
#SBATCH --ntasks=3
#SBATCH --nodes=3  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64           # 64 cores per rank
#SBATCH --mem=0                       # Request all available memory on each node
#SBATCH --qos=long

ulimit -c 0  # Disable core dumps

use_matrix_ops_petsc_mpi

echo "Environment:"
echo "  Rust: $(rustc --version)"
echo "  MPI: $(mpirun --version | head -n 1)"
echo "  SLEPC_DIR: $SLEPC_DIR"
echo "  PETSC_DIR: $PETSC_DIR"
echo ""

# Allow 64 threads per MPI rank
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# # # Intel MPI pinning - each rank gets 8 cores
# export I_MPI_PIN_DOMAIN=omp         # Pin to OMP domain
# export I_MPI_PIN=1

# Build with job-specific target directory and optimizations
echo "Building project in release mode for job ${SLURM_JOB_ID}..."
RUSTFLAGS="-C target-cpu=native" cargo build --release --target-dir ./target_${SLURM_JOB_ID}
if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi
echo "Build complete."
echo ""

# Run the job-specific binary
echo "Running job ${SLURM_JOB_ID}..."
echo "Number of MPI ranks: ${SLURM_NTASKS}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"

srun --cpu-bind=cores ./target_${SLURM_JOB_ID}/release/light-matter

echo "Job ${SLURM_JOB_ID} completed."

# Cleanup job-specific build directory
echo "Cleaning up build directory for job ${SLURM_JOB_ID}..."
rm -rf ./target_${SLURM_JOB_ID}