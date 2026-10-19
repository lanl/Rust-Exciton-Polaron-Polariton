#!/bin/bash
#SBATCH --job-name=band_gather
#SBATCH --output=logs/band_gather/band_gather_%j.out
#SBATCH --error=logs/band_gather/band_gather_%j.err
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

set -euo pipefail
module load gnuplot

source "/users/mike_tay/rust_exciton_pp/scripts/slurm/pipeline/pipeline_runtime.sh"
load_submitted_pipeline_config

# shellcheck disable=SC1090
source "${ENV_SCRIPT}"

# Set python plotting venv
source ~/venvs/rust-exciton-plotting/bin/activate

cd "${PROJECT_ROOT}"
print_submitted_pipeline_config

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Plotting configuration (all optional; may come from the submitted config
# or from the environment at submit time via --export=ALL)
# ---------------------------------------------------------------------------
PIPELINE_RUN_DIR="${PROJECT_ROOT}/pipeline_runs/${PIPELINE_RUN_ID}"
PLOT_DIR="${PIPELINE_RUN_DIR}/plots"
PLOTTING_DIR="${PROJECT_ROOT}/plotting"
PLOT_PYTHON="${PLOT_PYTHON:-python3}"
PLOT_ENV_SCRIPT="${PLOT_ENV_SCRIPT:-}"
BAND_PLOT_STRICT="${BAND_PLOT_STRICT:-0}"

mkdir -p "${PLOT_DIR}"

GATHER_LOG="${PIPELINE_RUN_DIR}/band_gather_${SLURM_JOB_ID}.log"
GATHER_MARKER="${PIPELINE_RUN_DIR}/.band_gather_start_${SLURM_JOB_ID}"
touch "${GATHER_MARKER}"

echo "Bandstructure gather"
echo "  SLURM_JOB_ID =${SLURM_JOB_ID}"
echo "  K_N_CHUNKS   = ${N_BAND_CHUNKS}"
echo "  PLOT_DIR     =${PLOT_DIR}"

# stdout is tee'd so the gathered filename can be read back from the log.
# pipefail ensures an srun failure still fails the job.
srun \
  --mpi=pmix_v5 \
  --ntasks=1 \
  --cpu-bind=cores \
  "${BINARY_PATH}" \
  --rayon-threads 1 \
  bandstructure \
  "${EXCITON_ARGS[@]}" \
  "${BAND_LM_ARGS[@]}" \
  "${BAND_SCAN_ARGS[@]}" \
  --method "${BAND_METHOD}" \
  --parallel gather-chunks \
  --k-n-chunks "${N_BAND_CHUNKS}" \
  | tee "${GATHER_LOG}"

# ---------------------------------------------------------------------------
# Locate the gathered file written by gather_color_band_structure_chunks
# ---------------------------------------------------------------------------
locate_gathered_band_file() {
    local path

    # 1) Exact path reported by Rust.
    path="$(sed -n 's/.*Saved gathered bandstructure to //p' "${GATHER_LOG}" | tail -n 1)"
    if [[ -n "${path}" && -f "${path}" ]]; then
        realpath "${path}"
        return 0
    fi

    # 2) Fallback: newest gathered (non-chunk, non-tmp) file written by this job.
    path="$(
        find pp_data -maxdepth 1 -type f -name 'color_band_a0*.bin' \
            -newer "${GATHER_MARKER}" -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | tail -n 1 | cut -d' ' -f2-
    )"
    if [[ -n "${path}" && -f "${path}" ]]; then
        realpath "${path}"
        return 0
    fi

    return 1
}

# ---------------------------------------------------------------------------
# Run the Python inspection + plotting scripts
# NOTE: called in an `if` context, so `set -e` is inactive here; every step
# checks its own status explicitly.
# ---------------------------------------------------------------------------
run_band_plots() {
    local band_file="$1"
    local stem
    stem="$(basename "${band_file}" .bin)"

    # Headless matplotlib; keep its cache off the (possibly read-only/slow) home FS.
    export MPLBACKEND=Agg
    export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig_${SLURM_JOB_ID}"
    mkdir -p "${MPLCONFIGDIR}" || return 1

    if [[ -n "${PLOT_ENV_SCRIPT}" ]]; then
        echo "Sourcing plotting environment: ${PLOT_ENV_SCRIPT}"
        # conda/module scripts often reference unset variables
        set +u
        # shellcheck disable=SC1090
        source "${PLOT_ENV_SCRIPT}" || { set -u; return 1; }
        set -u
    fi

    if ! command -v "${PLOT_PYTHON}" >/dev/null 2>&1; then
        echo "ERROR: plotting interpreter '${PLOT_PYTHON}' not found" >&2
        return 1
    fi

    if ! "${PLOT_PYTHON}" -c 'import numpy, matplotlib' 2>/dev/null; then
        echo "ERROR:${PLOT_PYTHON} lacks numpy/matplotlib (set PLOT_ENV_SCRIPT or PLOT_PYTHON)" >&2
        return 1
    fi

    echo "Plotting with $(command -v "${PLOT_PYTHON}") ($("${PLOT_PYTHON}" --version 2>&1))"

    # # Optional user arguments for plot_bands.py (array from the submitted config).
    # local extra_args=()
    # if declare -p BAND_PLOT_ARGS >/dev/null 2>&1; then
    #     extra_args=("${BAND_PLOT_ARGS[@]}")
    # fi

    echo "--- inspect_band_file.py ---"
    "${PLOT_PYTHON}" "${PLOTTING_DIR}/inspect_band_file.py" "${band_file}" \
        | tee "${PLOT_DIR}/${stem}_inspect.txt" \
        || return 1

    echo "--- plot_bands.py (PDF) ---"
    "${PLOT_PYTHON}" "${PLOTTING_DIR}/plot_bands.py" \
        "${band_file}" \
        -o "${PLOT_DIR}/${stem}.pdf" \${extra_args[@]+"${extra_args[@]}"} \
        || return 1

    echo "--- plot_bands.py (PNG preview) ---"
    "${PLOT_PYTHON}" "${PLOTTING_DIR}/plot_bands.py" \
        "${band_file}" \
        -o "${PLOT_DIR}/${stem}.png" \
        --dpi 300 \
        || return 1

    return 0
}

collect_rust_plots() {
    # The Rust gather also writes a gnuplot PDF into PROJECT_ROOT (if gnuplot exists).
    find . -maxdepth 1 -type f -name 'sing_ex_color_disp_plot_*.pdf' \
        -newer "${GATHER_MARKER}" -exec cp -p {} "${PLOT_DIR}/" \; 2>/dev/null || true
}

if ! BAND_FILE="$(locate_gathered_band_file)"; then
    echo "ERROR: gather finished but no gathered color_band_*.bin could be located" >&2
    rm -f "${GATHER_MARKER}"
    exit 1
fi

echo "Gathered band file: ${BAND_FILE}"
echo "${BAND_FILE}" > "${PIPELINE_RUN_DIR}/gathered_band_file.txt"

collect_rust_plots

if run_band_plots "${BAND_FILE}"; then
    echo "Band plots written to ${PLOT_DIR}:"
    ls -l "${PLOT_DIR}"
else
    echo "WARNING: Python band plotting failed (gathered data is intact: ${BAND_FILE})" >&2
    echo "Re-run manually with:" >&2
    echo "${PLOT_PYTHON} ${PLOTTING_DIR}/plot_bands.py${BAND_FILE} -o ${PLOT_DIR}/bands.pdf" >&2

    if [[ "${BAND_PLOT_STRICT}" == "1" ]]; then
        rm -f "${GATHER_MARKER}"
        exit 1
    fi
fi

rm -f "${GATHER_MARKER}"