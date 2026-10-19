#!/usr/bin/env bash

# Load the immutable pipeline configuration captured by sbatch at submission
# time. Jobs must not source pipeline_params.sh directly.
load_submitted_pipeline_config() {
    if [[ -z "${PIPELINE_CONFIG_B64:-}" ]]; then
        echo "ERROR: PIPELINE_CONFIG_B64 is not set." >&2
        echo "Submit this job through submit_pipelines.sh." >&2
        return 1
    fi

    local decoded
    decoded="$(printf '%s' "${PIPELINE_CONFIG_B64}" | base64 --decode)"

    if [[ -n "${PIPELINE_CONFIG_SHA256:-}" ]]; then
        local actual_sha256
        actual_sha256="$(printf '%s' "${decoded}" | sha256sum | awk '{print $1}')"

        if [[ "${actual_sha256}" != "${PIPELINE_CONFIG_SHA256}" ]]; then
            echo "ERROR: Submitted pipeline configuration checksum mismatch." >&2
            echo "Expected:${PIPELINE_CONFIG_SHA256}" >&2
            echo "Actual:   ${actual_sha256}" >&2
            return 1
        fi
    fi

    # The payload is generated locally by submit_pipelines.sh using %q.
    # shellcheck disable=SC1090
    source <(printf '%s\n' "${decoded}")

    require_pipeline_variable PROJECT_ROOT
    require_pipeline_variable BINARY_PATH
    require_pipeline_variable ENV_SCRIPT
    require_pipeline_variable CACHE_DIR
    require_pipeline_variable PIPELINE_RUN_ID
}

require_pipeline_variable() {
    local name="$1"

    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: Required submitted variable ${name} is empty." >&2
        return 1
    fi
}

print_submitted_pipeline_config() {
    echo "Submitted pipeline configuration:"
    echo "  PIPELINE_RUN_ID=${PIPELINE_RUN_ID}"
    echo "  PROJECT_ROOT=${PROJECT_ROOT}"
    echo "  BINARY_PATH=${BINARY_PATH}"
    echo "  ENV_SCRIPT=${ENV_SCRIPT}"
    echo "  CACHE_DIR=${CACHE_DIR}"
    echo "  N_BAND_CHUNKS=${N_BAND_CHUNKS}"
    echo "  REFLECTIVITIES_CSV=${REFLECTIVITIES_CSV}"
    echo "  BAND_METHOD=${BAND_METHOD}"
    echo "  PIPELINE_CONFIG_SHA256=${PIPELINE_CONFIG_SHA256:-unset}"
}