#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_PARAMS="${PIPELINE_DIR}/pipeline_params.sh"

# Read the editable parameter file exactly once, in the submission process.
# Submitted jobs never read this file.
# shellcheck disable=SC1090
source "${PIPELINE_PARAMS}"

WHAT="${1:-both}"

case "${WHAT}" in
    band|dielectric|both)
        ;;
    *)
        echo "Usage:$0 [band|dielectric|both]" >&2
        exit 2
        ;;
esac

cd "${PROJECT_ROOT}"

mkdir -p logs
mkdir -p "${CACHE_DIR}"
mkdir -p /lustre/scratch5/mike_tay/saved_data_lm

echo "Building light-matter once into ${TARGET_DIR}"

# shellcheck disable=SC1090
source "${ENV_SCRIPT}"

echo "Environment check:"
echo "  PROJECT_ROOT=${PROJECT_ROOT}"
echo "  TARGET_DIR=${TARGET_DIR}"
echo "  ENV_SCRIPT=${ENV_SCRIPT}"
echo "  PETSC_DIR=${PETSC_DIR:-unset}"
echo "  SLEPC_DIR=${SLEPC_DIR:-unset}"
echo "  mpicc=$(command -v mpicc)"
echo "  rustc=$(rustc --version)"

RUSTFLAGS="-C target-cpu=native" \
    cargo build --release --target-dir "${TARGET_DIR}"

BUILT_BINARY="${TARGET_DIR}/release/light-matter"

if [[ ! -x "${BUILT_BINARY}" ]]; then
    echo "ERROR: Expected binary does not exist: ${BUILT_BINARY}" >&2
    exit 1
fi

echo "Binary linkage check:"
ldd "${BUILT_BINARY}" \
    | grep -Ei 'mpi|pmix|petsc|slepc|open-pal|hwloc|event|blas|lapack' \
    || true

# Give this submission an immutable directory and binary.
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
nonce="$(printf '%06x' "$((RANDOM * RANDOM % 16777216))")"
export PIPELINE_RUN_ID="${timestamp}_${nonce}"

PIPELINE_RUN_DIR="${PROJECT_ROOT}/pipeline_runs/${PIPELINE_RUN_ID}"
mkdir -p "${PIPELINE_RUN_DIR}"

export BINARY_PATH="${PIPELINE_RUN_DIR}/light-matter"
install -m 0755 "${BUILT_BINARY}" "${BINARY_PATH}"

# Record a human-readable copy for provenance. Jobs do not load this file.
cp "${PIPELINE_PARAMS}" "${PIPELINE_RUN_DIR}/pipeline_params.original.sh"

IFS=',' read -r -a REFLECTIVITY_ARRAY <<< "${REFLECTIVITIES_CSV}"
N_REFLECTIVITIES="${#REFLECTIVITY_ARRAY[@]}"

if (( N_REFLECTIVITIES == 0 )); then
    echo "ERROR: REFLECTIVITIES_CSV contains no values." >&2
    exit 1
fi

if (( N_BAND_CHUNKS == 0 )); then
    echo "ERROR: N_BAND_CHUNKS must be positive." >&2
    exit 1
fi

append_scalar() {
    local name="$1"
    printf '%s=%q\n' "${name}" "${!name}"
}

append_array() {
    local name="$1"
    local -n values="${name}"

    printf '%s=(' "${name}"
    printf ' %q' "${values[@]}"
    printf ' )\n'
}

build_pipeline_payload() {
    append_scalar PROJECT_ROOT
    append_scalar TARGET_DIR
    append_scalar BINARY_PATH
    append_scalar ENV_SCRIPT
    append_scalar CACHE_DIR
    append_scalar PIPELINE_RUN_ID
    append_scalar N_BAND_CHUNKS
    append_scalar REFLECTIVITIES_CSV
    append_scalar BAND_METHOD

    append_array EXCITON_ARGS
    append_array BAND_LM_ARGS
    append_array BAND_SCAN_ARGS
    append_array DIELECTRIC_ARGS
}

PIPELINE_CONFIG="$(build_pipeline_payload)"
export PIPELINE_CONFIG_B64="$(
    printf '%s' "${PIPELINE_CONFIG}" | base64 | tr -d '\n'
)"
export PIPELINE_CONFIG_SHA256="$(
    printf '%s' "${PIPELINE_CONFIG}" | sha256sum | awk '{print$1}'
)"

printf '%s\n' "${PIPELINE_CONFIG}" \
    > "${PIPELINE_RUN_DIR}/submitted_config.sh"
printf '%s\n' "${PIPELINE_CONFIG_SHA256}" \
    > "${PIPELINE_RUN_DIR}/submitted_config.sha256"

# Supplying names rather than NAME=value avoids parsing problems and asks
# sbatch to capture the current exported values.
SBATCH_EXPORT='ALL,PIPELINE_CONFIG_B64,PIPELINE_CONFIG_SHA256'

submit_band() {
    echo "Submitting bandstructure pipeline with ${N_BAND_CHUNKS} chunks"

    local band_mat_job
    local band_chunks_job
    local band_gather_job

    band_mat_job="$(
        sbatch --parsable \
            --export="${SBATCH_EXPORT}" \
            --array="0-$((N_BAND_CHUNKS - 1))" \
            "${PIPELINE_DIR}/band_matcalc.sbatch"
    )"

    band_chunks_job="$(
        sbatch --parsable \
            --export="${SBATCH_EXPORT}" \
            --dependency="afterok:${band_mat_job}" \
            --array="0-$((N_BAND_CHUNKS - 1))" \
            "${PIPELINE_DIR}/band_chunks.sbatch"
    )"

    band_gather_job="$(
        sbatch --parsable \
            --export="${SBATCH_EXPORT}" \
            --dependency="afterok:${band_chunks_job}" \
            "${PIPELINE_DIR}/band_gather.sbatch"
    )"

    echo "Band matrix-element job array: ${band_mat_job}"
    echo "Band chunk job array:          ${band_chunks_job}"
    echo "Band gather job:               ${band_gather_job}"
}

submit_dielectric() {
    echo "Submitting dielectric pipeline with${N_REFLECTIVITIES} reflectivity jobs"

    local diel_mat_job
    local diel_calc_job
    local diel_gather_job

    diel_mat_job="$(
        sbatch --parsable \
            --export="${SBATCH_EXPORT}" \
            --array="0-$((N_REFLECTIVITIES - 1))" \
            "${PIPELINE_DIR}/dielectric_matcalc.sbatch"
    )"

    diel_calc_job="$(
        sbatch --parsable \
            --export="${SBATCH_EXPORT}" \
            --dependency="afterok:${diel_mat_job}" \
            --array="0-$((N_REFLECTIVITIES - 1))" \
            "${PIPELINE_DIR}/dielectric_calc.sbatch"
    )"

    diel_gather_job="$(
        sbatch --parsable \
            --export="${SBATCH_EXPORT}" \
            --dependency="afterok:${diel_calc_job}" \
            "${PIPELINE_DIR}/dielectric_gather.sbatch"
    )"

    echo "Dielectric matrix-element job array: ${diel_mat_job}"
    echo "Dielectric calculation job array:    ${diel_calc_job}"
    echo "Dielectric gather/plot job:          ${diel_gather_job}"
}

echo "Pipeline run ID:       ${PIPELINE_RUN_ID}"
echo "Snapshot directory:    ${PIPELINE_RUN_DIR}"
echo "Configuration SHA-256: ${PIPELINE_CONFIG_SHA256}"

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
esac