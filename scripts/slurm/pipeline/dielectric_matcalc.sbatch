#!/bin/bash
#SBATCH --job-name=diel_matcalc
#SBATCH --output=logs/diel_matcalc_%A_%a.out
#SBATCH --error=logs/diel_matcalc_%A_%a.err
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

export REFLECTIVITY_INDEX="${SLURM_ARRAY_TASK_ID}"

# Matrix-element precompute uses one process with many Rayon threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Dielectric matrix-element precompute"
echo "  REFLECTIVITY_INDEX =${REFLECTIVITY_INDEX}"
echo "  RAYON_NUM_THREADS  = ${RAYON_NUM_THREADS}"
echo "  CACHE_DIR          =${CACHE_DIR}"

"${TARGET_DIR}/release/light-matter" \
  --rayon-threads "${SLURM_CPUS_PER_TASK}" \
  matrix-elem-calc dielectric \
  "${EXCITON_ARGS[@]}" \
  "${DIELECTRIC_ARGS[@]}" \
  --reflectivity-index "${REFLECTIVITY_INDEX}" \
  --cache-dir "${CACHE_DIR}"