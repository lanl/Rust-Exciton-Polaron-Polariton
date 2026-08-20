#!/bin/bash
#SBATCH --job-name=dielectric
#SBATCH --output=logs/dielectric_%A_%a.out
#SBATCH --error=logs/dielectric_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G
#SBATCH --qos=long

set -euo pipefail

: "${PIPELINE_PARAMS:?PIPELINE_PARAMS was not set. Submit through submit_pipelines.sh or pass --export=ALL,PIPELINE_PARAMS=/path/to/pipeline_params.sh}"

source "${PIPELINE_PARAMS}"
source "${ENV_SCRIPT}"

cd "${PROJECT_ROOT}"

export RUST_BACKTRACE=full
export REFLECTIVITY_INDEX="${SLURM_ARRAY_TASK_ID}"

# Distributed SLEPc shell backend: many MPI ranks, one thread each.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

echo "Dielectric calculation"
echo "  REFLECTIVITY_INDEX =${REFLECTIVITY_INDEX}"
echo "  SLURM_NTASKS       = ${SLURM_NTASKS}"

export PMIX_MCA_psec=native
export PRTE_MCA_psec=native

srun \
  --mpi=pmix_v6 \
  --ntasks="${SLURM_NTASKS}" \
  --cpu-bind=cores \
  "${TARGET_DIR}/release/light-matter" \
  dielectric \
  "${EXCITON_ARGS[@]}" \
  "${DIELECTRIC_ARGS[@]}" \
  --reflectivity-index "${REFLECTIVITY_INDEX}"