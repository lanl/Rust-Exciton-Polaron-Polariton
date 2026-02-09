#!/bin/bash
#
#SBATCH --job-name=sing_ex_disp
#SBATCH --output myjob_%j.out
#SBATCH --error myjob_%j.err
#SBATCH --partition=general
#
#SBATCH --time=2-00:00:00
###SBATCH --time=8:00:00
#SBATCH --ntasks=3
#SBATCH --nodes=3  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64           # 64 cores per rank
#SBATCH --mem-per-cpu=3G             # 3GB per CPU (192GB total per node)
#SBATCH --qos=long

ulimit -c 0  # Disable core dumps

# Load Intel MKL
module load intel-mkl
module load intel-mpi

# Allow 8 threads per MPI rank
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# # Intel MPI pinning - each rank gets 8 cores
export I_MPI_PIN_DOMAIN=omp         # Pin to OMP domain
export I_MPI_PIN=1

# Build with job-specific target directory to avoid bus errors from concurrent builds
echo "Building binary for job ${SLURM_JOB_ID}..."
cargo build --release --target-dir ./target_${SLURM_JOB_ID}

# Run the job-specific binary
echo "Running job ${SLURM_JOB_ID}..."
echo "Number of MPI ranks: ${SLURM_NTASKS}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"

srun ./target_${SLURM_JOB_ID}/release/light-matter

echo "Job ${SLURM_JOB_ID} completed."

# Cleanup job-specific build directory
echo "Cleaning up build directory for job ${SLURM_JOB_ID}..."
rm -rf ./target_${SLURM_JOB_ID}