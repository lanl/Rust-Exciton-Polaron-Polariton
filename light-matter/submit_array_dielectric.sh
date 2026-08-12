#!/bin/bash
#SBATCH --job-name=dielectric_dist_shell
#SBATCH --output=logs/dielectric_%A_%a.out
#SBATCH --error=logs/dielectric_%A_%a.err
#SBATCH --partition=skylake-gold
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --qos=long
#SBATCH --array=0-2

set -euo pipefail

source "${HOME}/env/matrix_ops_petsc_mpi.sh"

# use_matrix_ops_petsc_mpi

export REFLECTIVITY_INDEX=${SLURM_ARRAY_TASK_ID}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

TARGET_DIR="./target_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

RUSTFLAGS="-C target-cpu=native" cargo build --release --target-dir "${TARGET_DIR}"

mpirun -np "${SLURM_NTASKS}" --bind-to core \
    "${TARGET_DIR}/release/light-matter"

rm -rf "${TARGET_DIR}"