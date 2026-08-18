#!/bin/bash
#SBATCH --job-name=band_chunks
#SBATCH --output=logs/band_%A_%a.out
#SBATCH --error=logs/band_%A_%a.err
#SBATCH --partition=skylake-gold
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
####SBATCH --qos=long
#SBATCH --array=0-15

# set -euo pipefail

export RUST_BACKTRACE=full

source "${HOME}/env/matrix_ops_petsc_mpi.sh"

export K_N_CHUNKS=16

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

TARGET_DIR="/vast/home/mike_tay/two_e_polaron-polariton/rust_sub/target"

# RUSTFLAGS="-C target-cpu=native" cargo build --release --target-dir "${TARGET_DIR}"

mpirun -np "${SLURM_NTASKS}" \
    --use-hwthread-cpus \
    --bind-to hwthread \
    "${TARGET_DIR}/release/light-matter" \
    bandstructure \
    --method linop \
    --parallel distributed-chunks \
    --k-chunk-index "${SLURM_ARRAY_TASK_ID}" \
    --k-n-chunks "${K_N_CHUNKS}" \
    --n-k-points 128 \
    --n-eig 128 \
    --n-bands 16 \
    --n-rel 16 \
    --n-ex-eig 32 \
    --total-coupling 0.003 \
    --omega-0 0.15 \
    --n-modes 11 \
    --q-max 0.001 \
    --pn-coupling 0.0 \
    --pn-n-fock 0 

rm -rf "${TARGET_DIR}"