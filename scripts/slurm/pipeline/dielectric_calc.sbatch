#!/bin/bash
#SBATCH --job-name=dielectric
#SBATCH --output=logs/dielectric_%A_%a.out
#SBATCH --error=logs/dielectric_%A_%a.err
#SBATCH --partition=standard
#SBATCH --time=16:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G

set -euo pipefail

: "${PIPELINE_PARAMS:?PIPELINE_PARAMS was not set. Submit through submit_pipelines.sh}"

source "${PIPELINE_PARAMS}"
source "${ENV_SCRIPT}"

cd "${PROJECT_ROOT}"

export RUST_BACKTRACE=full
export REFLECTIVITY_INDEX="${SLURM_ARRAY_TASK_ID}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

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
echo "  SLURM_JOB_NODELIST   =${SLURM_JOB_NODELIST}"
echo "  SLURM_NNODES         = ${SLURM_NNODES}"
echo "  SLURM_NTASKS         =${SLURM_NTASKS}"
echo "  SLURM_TASKS_PER_NODE = ${SLURM_TASKS_PER_NODE}"
echo "  PETSC_DIR            =${PETSC_DIR}"
echo "  SLEPC_DIR            = ${SLEPC_DIR}"
echo "  mpicc                =$(command -v mpicc)"

echo "Testing basic Slurm placement:"
srun --mpi=none -N "${SLURM_NNODES}" -n "${SLURM_NNODES}" hostname

echo "Testing PMIx v5 placement:"
run_pmix_v5 \
  -N "${SLURM_NNODES}" \
  -n "${SLURM_NNODES}" \
  hostname

echo "Starting dielectric solve"

run_pmix_v5 \
  --ntasks="${SLURM_NTASKS}" \
  --cpu-bind=cores \
  "${TARGET_DIR}/release/light-matter" \
  --rayon-threads 1 \
  dielectric \
  "${EXCITON_ARGS[@]}" \
  "${DIELECTRIC_ARGS[@]}" \
  --reflectivity-index "${REFLECTIVITY_INDEX}"