#!/bin/bash
#SBATCH --job-name=band_gather
#SBATCH --output=logs/band_gather_%j.out
#SBATCH --error=logs/band_gather_%j.err
#SBATCH --partition=skylake-gold
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

source "${HOME}/env/matrix_ops_petsc_mpi.sh"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

cargo build --release

./target/release/light-matter \
    bandstructure \
    --method sparse \
    --parallel gather-chunks \
    --k-n-chunks 16 \
    --n-k-points 128 \
    --n-eig 512 \
    --n-bands 16 \
    --n-rel 16 \
    --n-ex-eig 32 \
    --total-coupling 0.003 \
    --omega-0 0.15 \
    --n-modes 23 \
    --q-max 0.001 \
    --pn-coupling 0.0 \
    --pn-n-fock 0 \
    --zoomed true