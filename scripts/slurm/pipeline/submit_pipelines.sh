#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_PARAMS="${PIPELINE_DIR}/pipeline_params.sh"

source "${PIPELINE_PARAMS}"

cd "${PROJECT_ROOT}"
mkdir -p logs

echo "Building light-matter once into ${TARGET_DIR}"
source "${ENV_SCRIPT}"
RUSTFLAGS="-C target-cpu=native" cargo build --release --target-dir "${TARGET_DIR}"

IFS=',' read -r -a REFLECTIVITY_ARRAY <<< "${REFLECTIVITIES_CSV}"
N_REFLECTIVITIES="${#REFLECTIVITY_ARRAY[@]}"

WHAT="${1:-both}"

submit_band() {
  echo "Submitting bandstructure pipeline with ${N_BAND_CHUNKS} chunks"

  band_mat_job=$(
    sbatch --parsable \
      --export=ALL,PIPELINE_PARAMS="${PIPELINE_PARAMS}" \
      --array="0-$((N_BAND_CHUNKS - 1))" \
      "${PIPELINE_DIR}/band_matcalc.sbatch"
  )

  band_chunks_job=$(
    sbatch --parsable \
      --export=ALL,PIPELINE_PARAMS="${PIPELINE_PARAMS}" \
      --dependency="afterok:${band_mat_job}" \
      --array="0-$((N_BAND_CHUNKS - 1))" \
      "${PIPELINE_DIR}/band_chunks.sbatch"
  )

  band_gather_job=$(
    sbatch --parsable \
      --export=ALL,PIPELINE_PARAMS="${PIPELINE_PARAMS}" \
      --dependency="afterok:${band_chunks_job}" \
      "${PIPELINE_DIR}/band_gather.sbatch"
  )

  echo "Band matrix-element job array: ${band_mat_job}"
  echo "Band chunk job array:${band_chunks_job}"
  echo "Band gather job:               ${band_gather_job}"
}

submit_dielectric() {
  echo "Submitting dielectric pipeline with${N_REFLECTIVITIES} reflectivity jobs"

  diel_mat_job=$(
    sbatch --parsable \
      --export=ALL,PIPELINE_PARAMS="${PIPELINE_PARAMS}" \
      --array="0-$((N_REFLECTIVITIES - 1))" \
      "${PIPELINE_DIR}/dielectric_matcalc.sbatch"
  )

  diel_calc_job=$(
    sbatch --parsable \
      --export=ALL,PIPELINE_PARAMS="${PIPELINE_PARAMS}" \
      --dependency="afterok:${diel_mat_job}" \
      --array="0-$((N_REFLECTIVITIES - 1))" \
      "${PIPELINE_DIR}/dielectric_calc.sbatch"
  )

  echo "Dielectric matrix-element job array:${diel_mat_job}"
  echo "Dielectric calculation job array:    ${diel_calc_job}"
}

case "${WHAT}" in
  band)
    submit_band
    ;;
  dielectric)
    submit_dielectric
    ;;
  both)
    submit_band
    submit_dielectric
    ;;
  *)
    echo "Usage: $0 [band|dielectric|both]"
    exit 1
    ;;
esac