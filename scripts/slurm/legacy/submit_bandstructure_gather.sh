#!/bin/bash
#SBATCH --job-name=band_gather
#SBATCH --output=logs/band_gather_%j.out
#SBATCH --error=logs/band_gather_%j.err
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

source "${HOME}/env/matrix_ops_petsc_mpi.sh"

TARGET_DIR="/users/mike_tay/rust_exciton_pp/target"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

# cargo build --release

"${TARGET_DIR}/release/light-matter" \
    bandstructure \
    --method linop \
    --parallel gather-chunks \
    --k-n-chunks 4 \
    --n-k-points 128 \
    --n-eig 128 \
    --n-bands 16 \
    --n-rel 16 \
    --n-ex-eig 32 \
    --total-coupling 0.003 \
    --omega-0 0.15 \
    --photon-modes cartesian \
    --n-modes 11 \
    --q-max 0.001 \
    --pn-coupling 0.0 \
    --pn-n-fock 0 