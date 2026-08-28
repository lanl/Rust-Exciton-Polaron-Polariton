#!/bin/bash
#SBATCH --job-name=dielectric_dist_shell
#SBATCH --output=logs/dielectric_%A_%a.out
#SBATCH --error=logs/dielectric_%A_%a.err
#SBATCH --partition=standard
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
###SBATCH --qos=long
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

mpirun -np "${SLURM_NTASKS}" \
    --use-hwthread-cpus \
    --bind-to hwthread \
    "${TARGET_DIR}/release/light-matter" \
    dielectric \
    --photon-modes effective \
    --a0 16.0 \
    --v0 0.05 \
    --n-kappa 7 \
    --grid-size 64 \
    --box-size 3.0 \
    --n-rel 16 \
    --n-ex-eig 16 \
    --n-eig 1024 \
    --omega-0 0.18 \
    --omega-max 0.2 \
    --d-omega 1e-4 \
    --n-modes 32 \
    --reflectivities 0.0,0.999999,0.9999999 \
    --reflectivity-index "${SLURM_ARRAY_TASK_ID}" \
    --pn-coupling 0.0 \
    --pn-n-fock 1 \
    --omega-min-scan 0.0 \
    --omega-max-scan 0.3 \
    --n-omega-points 100000 \
    --output-dir dielectric_scan/effective_modes