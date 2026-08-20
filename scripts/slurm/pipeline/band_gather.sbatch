#!/bin/bash
#SBATCH --job-name=band_gather
#SBATCH --output=logs/band_gather_%j.out
#SBATCH --error=logs/band_gather_%j.err
#SBATCH --partition=skylake-gold
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail

: "${PIPELINE_PARAMS:?PIPELINE_PARAMS was not set. Submit through submit_pipelines.sh or pass --export=ALL,PIPELINE_PARAMS=/path/to/pipeline_params.sh}"

source "${PIPELINE_PARAMS}"
source "${ENV_SCRIPT}"

cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

echo "Gathering bandstructure chunks"
echo "  K_N_CHUNKS = ${N_BAND_CHUNKS}"

"${TARGET_DIR}/release/light-matter" \
  bandstructure \
  "${EXCITON_ARGS[@]}" \
  "${BAND_LM_ARGS[@]}" \
  "${BAND_SCAN_ARGS[@]}" \
  --method "${BAND_METHOD}" \
  --parallel gather-chunks \
  --k-n-chunks "${N_BAND_CHUNKS}" \
  --use-cached