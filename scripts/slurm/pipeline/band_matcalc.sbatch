#!/bin/bash
#SBATCH --job-name=band_matcalc
#SBATCH --output=logs/band_matcalc/band_matcalc_%A_%a.out
#SBATCH --error=logs/band_matcalc/band_matcalc_%A_%a.err
#SBATCH --partition=standard
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=500000

set -euo pipefail

source "/users/mike_tay/rust_exciton_pp/scripts/slurm/pipeline/pipeline_runtime.sh"
load_submitted_pipeline_config

# shellcheck disable=SC1090
source "${ENV_SCRIPT}"

cd "${PROJECT_ROOT}"
print_submitted_pipeline_config

export RUST_BACKTRACE=full

export K_N_CHUNKS="${N_BAND_CHUNKS}"
export K_CHUNK_INDEX="${SLURM_ARRAY_TASK_ID}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Band matrix-element chunk precompute"
echo "  SLURM_JOB_ID        = ${SLURM_JOB_ID}"
echo "  SLURM_ARRAY_TASK_ID =${SLURM_ARRAY_TASK_ID}"
echo "  K_CHUNK_INDEX       = ${K_CHUNK_INDEX}"
echo "  K_N_CHUNKS          =${K_N_CHUNKS}"
echo "  RAYON_NUM_THREADS   = ${RAYON_NUM_THREADS}"
echo "  CACHE_DIR           =${CACHE_DIR}"

"${BINARY_PATH}" \
  --rayon-threads "${SLURM_CPUS_PER_TASK}" \
  matrix-elem-calc \
  bandstructure \
  "${EXCITON_ARGS[@]}" \
  "${BAND_LM_ARGS[@]}" \
  "${BAND_SCAN_ARGS[@]}" \
  --cache-dir "${CACHE_DIR}" \
  --k-n-chunks "${K_N_CHUNKS}" \
  --k-chunk-index "${K_CHUNK_INDEX}"