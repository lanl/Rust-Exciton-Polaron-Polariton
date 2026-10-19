#!/bin/bash
#SBATCH --job-name=diel_gather
#SBATCH --output=logs/diel_gather/diel_gather_%j.out
#SBATCH --error=logs/diel_gather/diel_gather_%j.err
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

source "/users/mike_tay/rust_exciton_pp/scripts/slurm/pipeline/pipeline_runtime.sh"
load_submitted_pipeline_config

# shellcheck disable=SC1090
source "${ENV_SCRIPT}"

cd "${PROJECT_ROOT}"
print_submitted_pipeline_config

export RUST_BACKTRACE=full

DIEL_OUT_DIR="dielectric_scan/effective_modes"
for ((i = 0; i < ${#DIELECTRIC_ARGS[@]}; i++)); do
  case "${DIELECTRIC_ARGS[i]}" in
    --output-dir)
      if ((i + 1 < ${#DIELECTRIC_ARGS[@]})); then
        DIEL_OUT_DIR="${DIELECTRIC_ARGS[i + 1]}"
      fi
      ;;
    --output-dir=*)
      DIEL_OUT_DIR="${DIELECTRIC_ARGS[i]#--output-dir=}"
      ;;
  esac
done

# Single-process plotting/gather job. No distributed SLEPc solve here.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Dielectric gather/plot"
echo "  SLURM_JOB_ID       =${SLURM_JOB_ID}"
echo "  PROJECT_ROOT       = ${PROJECT_ROOT}"
echo "  TARGET_DIR         =${TARGET_DIR}"
echo "  OUTPUT_DIR         = ${DIEL_OUT_DIR}"
echo "  RAYON_NUM_THREADS  = ${RAYON_NUM_THREADS}"
echo "  PETSC_DIR          =${PETSC_DIR}"
echo "  SLEPC_DIR          = ${SLEPC_DIR}"

echo "Existing dielectric data files:"
find "${DIEL_OUT_DIR}" -maxdepth 1 -type f -name 'dielectric_*.dat' -print | sort || true

# Use calculate=false to trigger the plotting/gather branch in app/dielectric.rs.
# Do not pass --reflectivity-index here: this gather step plots all reflectivities.
"${BINARY_PATH}" \
  --rayon-threads "${SLURM_CPUS_PER_TASK}" \
  dielectric \
  "${EXCITON_ARGS[@]}" \
  "${DIELECTRIC_ARGS[@]}" \
  --calculate=false

echo "Generated dielectric plots:"
find "${DIEL_OUT_DIR}" -maxdepth 1 \
  \( -name '*.svg' -o -name '*.pdf' -o -name '*.png' \) \
  -print | sort || true
