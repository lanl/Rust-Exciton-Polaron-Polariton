#!/bin/bash
#SBATCH --job-name=band_gather
#SBATCH --output=logs/band_gather_%j.out
#SBATCH --error=logs/band_gather_%j.err
#SBATCH --partition=skylake-gold
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --array=0-2

set -euo pipefail

source "${HOME}/env/matrix_ops_petsc_mpi.sh"

cargo build --release

./target/release/light-matter \
    --rayon-threads 128 \
    matrix-elem-calc \
    --a0 16.0 \
    --v0 0.05 \
    --m-e 0.2 \
    --m-h 0.2 \
    --n-kappa 7 \
    --grid-size 64 \
    --box-size 3.0 \
    --save-dir saved_data_lm \
    --n-rel 16 \
    --n-ex-eig 16 \
    --omega-0 0.18 \
    --omega-max 0.2 \
    --d-omega 1e-4 \
    --n-modes 32 \
    --reflectivities 0.0,0.999999,0.9999999 \
    --reflectivity-index "${SLURM_ARRAY_TASK_ID}" \
    --cache-dir saved_data_lm/matrix_element_cache