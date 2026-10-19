#!/bin/bash
#SBATCH --job-name=band_chunks
#SBATCH --output=logs/band_chunk/band_%A_%a.out
#SBATCH --error=logs/band_chunk/band_%A_%a.err
#SBATCH --partition=standard
#SBATCH --time=16:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=500000

set -euo pipefail

source "/users/mike_tay/rust_exciton_pp/scripts/slurm/pipeline/pipeline_runtime.sh"
load_submitted_pipeline_config

# shellcheck disable=SC1090
source "${ENV_SCRIPT}"

cd "${PROJECT_ROOT}"
print_submitted_pipeline_config

export RUST_BACKTRACE=full

export K_N_CHUNKS="${N_BAND_CHUNKS}"
export K_CHUNK_INDEX="${SLURM_ARRAY_TASK_ID}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

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

echo "Bandstructure distributed chunk solve"
echo "  SLURM_JOB_ID         =${SLURM_JOB_ID}"
echo "  SLURM_JOB_NODELIST   = ${SLURM_JOB_NODELIST}"
echo "  SLURM_NNODES         =${SLURM_NNODES}"
echo "  SLURM_NTASKS         = ${SLURM_NTASKS}"
echo "  SLURM_TASKS_PER_NODE =${SLURM_TASKS_PER_NODE}"
echo "  K_CHUNK_INDEX        = ${K_CHUNK_INDEX}"
echo "  K_N_CHUNKS           =${K_N_CHUNKS}"

echo "Testing basic Slurm placement:"
srun --mpi=none -N "${SLURM_NNODES}" -n "${SLURM_NNODES}" hostname

echo "Testing PMIx v5 placement:"
run_pmix_v5 \
  -N "${SLURM_NNODES}" \
  -n "${SLURM_NNODES}" \
  hostname

echo "Starting bandstructure solve"

run_pmix_v5 \
  --ntasks="${SLURM_NTASKS}" \
  --cpu-bind=cores \
  "${BINARY_PATH}" \
  --rayon-threads "${SLURM_CPUS_PER_TASK}" \
  bandstructure \
  "${EXCITON_ARGS[@]}" \
  "${BAND_LM_ARGS[@]}" \
  "${BAND_SCAN_ARGS[@]}" \
  --method "${BAND_METHOD}" \
  --parallel distributed-chunks \
  --k-chunk-index "${K_CHUNK_INDEX}" \
  --k-n-chunks "${K_N_CHUNKS}"