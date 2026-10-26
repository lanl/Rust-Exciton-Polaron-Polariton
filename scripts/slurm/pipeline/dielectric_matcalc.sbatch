#!/bin/bash
#SBATCH --job-name=diel_matcalc
#SBATCH --output=logs/diel_matcalc/diel_matcalc_%A_%a.out
#SBATCH --error=logs/diel_matcalc/diel_matcalc_%A_%a.err
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=500000

set -euo pipefail

source "/users/mike_tay/rust_exciton_pp/scripts/slurm/pipeline/pipeline_runtime.sh"
load_submitted_pipeline_config

# shellcheck disable=SC1090
source "${ENV_SCRIPT}"

cd "${PROJECT_ROOT}"
print_submitted_pipeline_config

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

"${BINARY_PATH}" \
  --rayon-threads "${SLURM_CPUS_PER_TASK}" \
  matrix-elem-calc \
  dielectric \
  "${EXCITON_ARGS[@]}" \
  "${DIELECTRIC_ARGS[@]}" \
  --cache-dir "${CACHE_DIR}" \
  --reflectivity-index "${REFLECTIVITY_INDEX}"