#!/bin/bash
#SBATCH --job-name=band_chunks
#SBATCH --output=logs/band_chunks_%A_%a.out
#SBATCH --error=logs/band_chunks_%A_%a.err
#SBATCH --partition=skylake-gold
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=0

set -euo pipefail

: "${PIPELINE_PARAMS:?PIPELINE_PARAMS was not set. Submit through submit_pipelines.sh or pass --export=ALL,PIPELINE_PARAMS=/path/to/pipeline_params.sh}"

source "${PIPELINE_PARAMS}"
source "${ENV_SCRIPT}"

cd "${PROJECT_ROOT}"

export RUST_BACKTRACE=full

export K_N_CHUNKS="${N_BAND_CHUNKS}"
export K_CHUNK_INDEX="${SLURM_ARRAY_TASK_ID}"

# SLEPc/PETSc solve: MPI ranks, one thread each.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

echo "Bandstructure distributed chunk"
echo "  K_CHUNK_INDEX =${K_CHUNK_INDEX}"
echo "  K_N_CHUNKS    = ${K_N_CHUNKS}"
echo "  SLURM_NTASKS  =${SLURM_NTASKS}"

mpirun -np "${SLURM_NTASKS}" \
  --use-hwthread-cpus \
  --bind-to hwthread \
  "${TARGET_DIR}/release/light-matter" \
  bandstructure \
  "${EXCITON_ARGS[@]}" \
  "${BAND_LM_ARGS[@]}" \
  "${BAND_SCAN_ARGS[@]}" \
  --method "${BAND_METHOD}" \
  --parallel distributed-chunks \
  --k-n-chunks "${K_N_CHUNKS}" \
  --k-chunk-index "${K_CHUNK_INDEX}" \
  --use-cached