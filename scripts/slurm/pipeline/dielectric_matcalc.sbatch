#!/bin/bash
#SBATCH --job-name=diel_matcalc
#SBATCH --output=logs/diel_matcalc_%A_%a.out
#SBATCH --error=logs/diel_matcalc_%A_%a.err
#SBATCH --partition=standard
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=88
#SBATCH --mem=400G

set -euo pipefail

: "${PIPELINE_PARAMS:?PIPELINE_PARAMS was not set. Submit through submit_pipelines.sh}"

source "${PIPELINE_PARAMS}"
source "${ENV_SCRIPT}"

cd "${PROJECT_ROOT}"

export RUST_BACKTRACE=full
export REFLECTIVITY_INDEX="${SLURM_ARRAY_TASK_ID}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Dielectric matrix-element precompute"
echo "  SLURM_JOB_ID        =${SLURM_JOB_ID}"
echo "  SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "  REFLECTIVITY_INDEX  =${REFLECTIVITY_INDEX}"
echo "  RAYON_NUM_THREADS   = ${RAYON_NUM_THREADS}"
echo "  CACHE_DIR           =${CACHE_DIR}"

"${TARGET_DIR}/release/light-matter" \
  --rayon-threads "${SLURM_CPUS_PER_TASK}" \
  matrix-elem-calc \
  dielectric \
  "${EXCITON_ARGS[@]}" \
  "${DIELECTRIC_ARGS[@]}" \
  --cache-dir "${CACHE_DIR}" \
  --reflectivity-index "${REFLECTIVITY_INDEX}"