#!/bin/bash
#
#SBATCH --job-name=sing_ex_disp
#SBATCH --output myjob_%j.out
#SBATCH --error myjob_%j.err
#SBATCH --partition=standard
#
##SBATCH --time=8:00:00
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3G 
#SBATCH --qos=long

ulimit -c 0  # Disable core dumps

# Load Intel MKL
module load intel-mkl

# Build with job-specific target directory to avoid bus errors from concurrent builds
echo "Building binary for job ${SLURM_JOB_ID}..."
cargo run --release --target-dir ./target_${SLURM_JOB_ID}

echo "Job ${SLURM_JOB_ID} completed."

# Cleanup job-specific build directory
echo "Cleaning up build directory for job ${SLURM_JOB_ID}..."
rm -rf ./target_${SLURM_JOB_ID}