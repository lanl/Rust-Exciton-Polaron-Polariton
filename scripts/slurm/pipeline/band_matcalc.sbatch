#!/bin/bash
#SBATCH --job-name=band_matcalc
#SBATCH --output=logs/band_matcalc_%A_%a.out
#SBATCH --error=logs/band_matcalc_%A_%a.err
#SBATCH --partition=skylake-gold
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=88
#SBATCH --mem=0

set -euo pipefail

: "${PIPELINE_PARAMS:?PIPELINE_PARAMS was not set. Submit through submit_pipelines.sh or pass --export=ALL,PIPELINE_PARAMS=/path/to/pipeline_params.sh}"

source "${PIPELINE_PARAMS}"
source "${ENV_SCRIPT}"

cd "${PROJECT_ROOT}"

export RUST_BACKTRACE=full

export K_N_CHUNKS="${N_BAND_CHUNKS}"
export K_CHUNK_INDEX="${SLURM_ARRAY_TASK_ID}"

# Matrix-element precompute uses Rayon, not MPI.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Band matrix-element precompute"
echo "  K_CHUNK_INDEX     =${K_CHUNK_INDEX}"
echo "  K_N_CHUNKS        = ${K_N_CHUNKS}"
echo "  RAYON_NUM_THREADS =${RAYON_NUM_THREADS}"
echo "  CACHE_DIR         = ${CACHE_DIR}"

"${TARGET_DIR}/release/light-matter" \
  --rayon-threads "${SLURM_CPUS_PER_TASK}" \
  matrix-elem-calc bandstructure \
  "${EXCITON_ARGS[@]}" \
  "${BAND_LM_ARGS[@]}" \
  "${BAND_SCAN_ARGS[@]}" \
  --k-n-chunks "${K_N_CHUNKS}" \
  --k-chunk-index "${K_CHUNK_INDEX}" \
  --cache-dir "${CACHE_DIR}"