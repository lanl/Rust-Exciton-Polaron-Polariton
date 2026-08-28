#!/bin/bash
#SBATCH --job-name=band_gather
#SBATCH --output=logs/band_gather_%j.out
#SBATCH --error=logs/band_gather_%j.err
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

set -euo pipefail

: "${PIPELINE_PARAMS:?PIPELINE_PARAMS was not set. Submit through submit_pipelines.sh}"

source "${PIPELINE_PARAMS}"
source "${ENV_SCRIPT}"

cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

echo "Bandstructure gather"
echo "  SLURM_JOB_ID =${SLURM_JOB_ID}"
echo "  K_N_CHUNKS   = ${N_BAND_CHUNKS}"

srun \
  --mpi=pmix_v5 \
  --ntasks=1 \
  --cpu-bind=cores \
  "${TARGET_DIR}/release/light-matter" \
  --rayon-threads 1 \
  bandstructure \
  "${EXCITON_ARGS[@]}" \
  "${BAND_LM_ARGS[@]}" \
  "${BAND_SCAN_ARGS[@]}" \
  --method "${BAND_METHOD}" \
  --parallel gather-chunks \
  --k-n-chunks "${N_BAND_CHUNKS}"