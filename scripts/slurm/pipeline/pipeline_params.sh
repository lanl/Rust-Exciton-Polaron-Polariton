#!/usr/bin/env bash

# -------------------------------------------------------------------
# Project / environment
# -------------------------------------------------------------------
export PROJECT_ROOT="/users/mike_tay/rust_exciton_pp"
export TARGET_DIR="${PROJECT_ROOT}/target_pipeline"
export ENV_SCRIPT="${HOME}/env/matrix_ops_petsc_mpi_chicoma.sh"

# Must match PPConfig default unless you add --cache-dir support to normal
# bandstructure/dielectric commands. Matrix-element precompute can take this,
# but bandstructure/dielectric currently use PPConfig default.
export CACHE_DIR="/lustre/scratch5/mike_tay/saved_data_lm/matrix_element_cache"

# -------------------------------------------------------------------
# Pipeline parallelism
# -------------------------------------------------------------------
export N_BAND_CHUNKS=4

# Reflectivities for dielectric scan. Slurm array size is inferred from this.
export REFLECTIVITIES_CSV="0.0,0.999999,0.9999999"

# -------------------------------------------------------------------
# Shared exciton/model parameters
# Used by BOTH precompute and solve jobs.
# -------------------------------------------------------------------
EXCITON_ARGS=(
  --a0 16.0
  --v0 0.05
  --n-kappa 7
  --m-e 0.2
  --m-h 0.2
  --grid-size 64
  --box-size 3.0
  --save-dir /lustre/scratch5/mike_tay/saved_data_lm
  --batch-size 32
)

# -------------------------------------------------------------------
# Bandstructure parameters
# Split into physical/light-matter args and band-scan args, but both are
# reused by matcalc/chunks/gather.
# -------------------------------------------------------------------
BAND_LM_ARGS=(
  --photon-modes effective
  --n-rel 16
  --n-ex-eig 32
  --omega-0 0.18
  --omega-max 0.2
  --d-omega 2e-4
  --reflectivity 0.999999
  --photon-coupling-scale 5.0
  --n-modes 16
  --pn-coupling 0.0
  --pn-n-fock 0
)

BAND_SCAN_ARGS=(
  --n-k-points 128
  --n-eig 512
  --n-bands 16
)

# Backend used for the actual bandstructure solve.
export BAND_METHOD="linop"

# -------------------------------------------------------------------
# Dielectric parameters
# Used by BOTH dielectric matcalc and dielectric solve.
# -------------------------------------------------------------------
DIELECTRIC_ARGS=(
  --photon-modes effective
  --n-rel 16
  --n-ex-eig 16
  --n-eig 2048
  --omega-0 0.18
  --omega-max 0.2
  --d-omega 5e-5
  --n-modes 32
  --reflectivities "${REFLECTIVITIES_CSV}"
  --photon-coupling-scale 5.0
  --pn-coupling 0.0
  --pn-omega 0.0025
  --pn-k-x 0.05
  --pn-n-fock 1
  --omega-min-scan 0.0
  --omega-max-scan 0.3
  --n-omega-points 100000
  --output-dir dielectric_scan/effective_modes
)