#!/bin/bash
#SBATCH --job-name=dielectric
#SBATCH --output=logs/dielectric/dielectric_%A_%a.out
#SBATCH --error=logs/dielectric/dielectric_%A_%a.err
#SBATCH --partition=standard
#SBATCH --time=5-00:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=500000
#SBATCH --qos=long

set -euo pipefail

source "/users/mike_tay/rust_exciton_pp/scripts/slurm/pipeline/pipeline_runtime.sh"
load_submitted_pipeline_config

# shellcheck disable=SC1090
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

run_pmix_v5 --ntasks="${SLURM_NTASKS}" --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --cpu-bind=cores bash -c 'echo "rank$SLURM_PROCID on $(hostname): nproc=$(nproc) $(grep Cpus_allowed_list /proc/self/status)"'

echo "Dielectric calculation"
echo "  REFLECTIVITY_INDEX   = ${REFLECTIVITY_INDEX}"
echo "  SLURM_JOB_ID         = ${SLURM_JOB_ID}"
echo "  SLURM_JOB_NODELIST   = ${SLURM_JOB_NODELIST}"
echo "  SLURM_NNODES         = ${SLURM_NNODES}"
echo "  SLURM_NTASKS         = ${SLURM_NTASKS}"
echo "  SLURM_TASKS_PER_NODE = ${SLURM_TASKS_PER_NODE}"
echo "  SLURM_CPUS_PER_TASK  = ${SLURM_CPUS_PER_TASK}"
echo "  PETSC_DIR            = ${PETSC_DIR}"
echo "  SLEPC_DIR            = ${SLEPC_DIR}"
echo "  mpicc                = $(command -v mpicc)"

# echo "Testing basic Slurm placement:"
# srun --mpi=none -N "${SLURM_NNODES}" -n "${SLURM_NNODES}" hostname

# echo "Testing PMIx v5 placement:"
# run_pmix_v5 \
#   -N "${SLURM_NNODES}" \
#   -n "${SLURM_NNODES}" \
#   hostname

echo "Starting dielectric solve"

run_pmix_v5 \
  --ntasks="${SLURM_NTASKS}" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --cpu-bind=verbose,cores \
  "${BINARY_PATH}" \
  --rayon-threads "${SLURM_CPUS_PER_TASK}" \
  dielectric \
  "${EXCITON_ARGS[@]}" \
  "${DIELECTRIC_ARGS[@]}" \
  --reflectivity-index "${REFLECTIVITY_INDEX}"