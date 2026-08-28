#!/bin/bash
#SBATCH --job-name=diel_gather
#SBATCH --output=logs/diel_gather_%j.out
#SBATCH --error=logs/diel_gather_%j.err
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

: "${PIPELINE_PARAMS:?PIPELINE_PARAMS was not set. Submit through submit_pipelines.sh}"

source "${PIPELINE_PARAMS}"
source "${ENV_SCRIPT}"

cd "${PROJECT_ROOT}"

export RUST_BACKTRACE=full

# Single-process plotting/gather job. No distributed SLEPc solve here.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Dielectric gather/plot"
echo "  SLURM_JOB_ID       =${SLURM_JOB_ID}"
echo "  PROJECT_ROOT       = ${PROJECT_ROOT}"
echo "  TARGET_DIR         =${TARGET_DIR}"
echo "  OUTPUT_DIR         = dielectric_scan/effective_modes"
echo "  RAYON_NUM_THREADS  = ${RAYON_NUM_THREADS}"
echo "  PETSC_DIR          =${PETSC_DIR}"
echo "  SLEPC_DIR          = ${SLEPC_DIR}"

echo "Existing dielectric data files:"
find dielectric_scan/effective_modes -maxdepth 1 -type f -name 'dielectric_*.dat' -print | sort || true

# Use calculate=false to trigger the plotting/gather branch in app/dielectric.rs.
# Do not pass --reflectivity-index here: this gather step plots all reflectivities.
"${TARGET_DIR}/release/light-matter" \
  --rayon-threads "${SLURM_CPUS_PER_TASK}" \
  dielectric \
  "${EXCITON_ARGS[@]}" \
  "${DIELECTRIC_ARGS[@]}" \
  --calculate=false

echo "Generated dielectric plots:"
find dielectric_scan/effective_modes -maxdepth 1 \
  \( -name '*.svg' -o -name '*.pdf' -o -name '*.png' \) \
  -print | sort || true
