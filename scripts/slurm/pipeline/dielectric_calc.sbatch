#!/bin/bash
#SBATCH --job-name=dielectric
#SBATCH --output=logs/dielectric/dielectric_%A_%a.out
#SBATCH --error=logs/dielectric/dielectric_%A_%a.err
#SBATCH --open-mode=append          # keep logs across restarts (default truncates)
#SBATCH --partition=standard
#SBATCH --time=16:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=500000
#SBATCH --requeue                   # make the job eligible for requeue
#SBATCH --signal=B:USR1@300         # USR1 to the batch shell 5 min before the limit

set -euo pipefail

# Maximum automatic restarts per array task (override at submit time if needed).
MAX_RESTARTS="${MAX_RESTARTS:-5}"
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"

# Array tasks must be requeued by their own ID, not the array master ID.
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  REQUEUE_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
  REQUEUE_ID="${SLURM_JOB_ID}"
fi

SOLVER_PID=""

on_timeout_warning() {
  echo "[$(date)] Time-limit warning received (restart ${RESTART_COUNT}/${MAX_RESTARTS})"
  if (( RESTART_COUNT < MAX_RESTARTS )); then
    echo "[$(date)] Requeueing${REQUEUE_ID}; work resumes from the last checkpointed batch"
    scontrol requeue "${REQUEUE_ID}"
    # scontrol requeue terminates this allocation; nothing after this matters.
  else
    echo "[$(date)] MAX_RESTARTS reached; letting the job time out" >&2
  fi
  [[ -n "${SOLVER_PID}" ]] && kill -TERM "${SOLVER_PID}" 2>/dev/null || true
  exit 99
}
trap on_timeout_warning USR1

echo "========== start: restart count ${RESTART_COUNT} on$(hostname) at $(date) =========="

source "/users/mike_tay/rust_exciton_pp/scripts/slurm/pipeline/pipeline_runtime.sh"
load_submitted_pipeline_config

source "${ENV_SCRIPT}"

cd "${PROJECT_ROOT}"
print_submitted_pipeline_config

export RUST_BACKTRACE=full
export REFLECTIVITY_INDEX="${SLURM_ARRAY_TASK_ID}"

export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

export SLURM_PMIX_DIRECT_CONN=0

APP_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
APP_PATH="${PATH}"

run_pmix_v5() {
  env -u LD_LIBRARY_PATH \
    srun \
      --mpi=pmix_v5 \
      --export=ALL,LD_LIBRARY_PATH="${APP_LD_LIBRARY_PATH}",PATH="${APP_PATH}",SLURM_PMIX_DIRECT_CONN=0 \
      "$@"
}

echo "Dielectric calculation"
echo "  REFLECTIVITY_INDEX   =${REFLECTIVITY_INDEX}"
echo "  SLURM_JOB_ID         = ${SLURM_JOB_ID}"
echo "  REQUEUE_ID           =${REQUEUE_ID}"
echo "  SLURM_RESTART_COUNT  = ${RESTART_COUNT}"
echo "  SLURM_JOB_NODELIST   =${SLURM_JOB_NODELIST}"
echo "  SLURM_NTASKS         = ${SLURM_NTASKS}"

echo "Starting dielectric solve"

# Run srun in the background and `wait` on it. Bash only runs a trap between
# commands, so a foreground srun would delay the USR1 handler until srun exits,
# which is too late.
run_pmix_v5 \
  --ntasks="${SLURM_NTASKS}" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --cpu-bind=verbose,cores \
  "${BINARY_PATH}" \
  --rayon-threads "${SLURM_CPUS_PER_TASK}" \
  dielectric \
  "${EXCITON_ARGS[@]}" \
  "${DIELECTRIC_ARGS[@]}" \
  --reflectivity-index "${REFLECTIVITY_INDEX}" &
SOLVER_PID=$!

set +e
wait "${SOLVER_PID}"
status=$?
set -e

echo "[$(date)] Solver exited with status ${status}"
exit "${status}"