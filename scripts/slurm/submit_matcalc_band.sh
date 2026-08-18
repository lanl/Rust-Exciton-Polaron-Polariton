#!/bin/bash
#SBATCH --job-name=band_matcalc
#SBATCH --output=logs/band_matcalc_%A_%a.out
#SBATCH --error=logs/band_matcalc_%A_%a.err
#SBATCH --partition=skylake-gold
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=88
#SBATCH --array=0-3

# set -euo pipefail

export RUST_BACKTRACE=full

source "${HOME}/env/matrix_ops_petsc_mpi.sh"

TARGET_DIR="/vast/home/mike_tay/two_e_polaron-polariton/rust_sub/target"

export K_N_CHUNKS=4
export K_CHUNK_INDEX="${SLURM_ARRAY_TASK_ID}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Matrix-element chunk precompute"
echo "  SLURM_ARRAY_TASK_ID =${SLURM_ARRAY_TASK_ID}"
echo "  K_CHUNK_INDEX       = ${K_CHUNK_INDEX}"
echo "  K_N_CHUNKS          =${K_N_CHUNKS}"
echo "  RAYON_NUM_THREADS   = ${RAYON_NUM_THREADS}"

"${TARGET_DIR}/release/light-matter" \
    --rayon-threads "${SLURM_CPUS_PER_TASK}" \
    matrix-elem-calc bandstructure \
    --k-n-chunks "${K_N_CHUNKS}" \
    --k-chunk-index "${K_CHUNK_INDEX}" \
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