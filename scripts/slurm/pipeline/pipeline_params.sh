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
export N_BAND_CHUNKS=128

# Reflectivities for dielectric scan. Slurm array size is inferred from this.
# export REFLECTIVITIES_CSV="0.0,0.99999,0.999999"
export REFLECTIVITIES_CSV="0.0,0.992146025,0.9992146025" # Q = 0, 200, 2000

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
# BAND_LM_ARGS=(
#   --photon-modes effective
#   --n-rel 16
#   --n-ex-eig 32
#   --omega-0 0.16
#   --omega-max 0.195
#   --d-omega 4e-4
#   --reflectivity 0.9992146025
#   --photon-coupling-scale 250.0
#   --n-modes 32
#   --pn-coupling 0.0
#   --pn-n-fock 0
#   --zoomed
# )

BAND_LM_ARGS=(
  --photon-modes effective
  --n-rel 16
  --n-ex-eig 32
  --omega-0 0.16
  --omega-max 0.246
  --d-omega 4e-4
  --reflectivity 0.992146025
  --photon-coupling-scale 250.0
  --n-modes 21
  --pn-coupling 0.0
  --pn-n-fock 0
)

# BAND_LM_ARGS=(
#   --photon-modes effective
#   --n-rel 16
#   --n-ex-eig 32
#   --omega-0 0.16
#   --omega-max 0.21
#   --d-omega 4e-4
#   --reflectivity 0.992146025
#   --photon-coupling-scale 250.0
#   --n-modes 16
#   --pn-coupling 0.0
#   --pn-n-fock 0
#   --zoomed
# )

BAND_SCAN_ARGS=(
  --n-k-points 1024
  --n-eig 512              # with a cutoff: only the initial batch-size hint
  --n-bands 16
  --energy-cutoff=-0.01   # all states with E <= this; also the plot y-max
)

# Backend used for the actual bandstructure solve.
export BAND_METHOD="linop"

# -------------------------------------------------------------------
# Dielectric parameters
# Used by BOTH dielectric matcalc and dielectric solve.
# Sheet2D runs should use --output-dir dielectric_scan/sheet in a copied params file.
# -------------------------------------------------------------------
# DIELECTRIC_ARGS=(
#   --photon-modes effective
#   --n-rel 16
#   --n-ex-eig 16
#   --energy-cutoff 0.04
#   --omega-0 0.16
#   --omega-max 0.26
#   --d-omega 2e-4
#   --n-modes 45
#   --reflectivities "${REFLECTIVITIES_CSV}"
#   --photon-coupling-scale 250.0
#   --pn-coupling 1e-3
#   --pn-omega 0.0025
#   --pn-k-x 0.05
#   --pn-n-fock 1
#   --omega-min-scan 0.0
#   --omega-max-scan 0.3
#   --n-omega-points 100000
#   --output-dir dielectric_scan/slab
#   --dielectric-method slab
#   --exciton-density 1e-5
#   --slab-thickness-bohr 10
#   --eps-monitor
# )

DIELECTRIC_ARGS=(
  --photon-modes effective
  --n-rel 16
  --n-ex-eig 16
  --energy-cutoff 0.04
  --omega-0 0.16
  --omega-max 0.21
  --d-omega 2e-4
  --n-modes 45
  --reflectivities "${REFLECTIVITIES_CSV}"
  --photon-coupling-scale 250.0
  --pn-coupling 1e-3
  --pn-omega 0.0025
  --pn-k-x 0.05
  --pn-n-fock 4
  --omega-min-scan 0.0
  --omega-max-scan 0.3
  --n-omega-points 100000
  --output-dir dielectric_scan/sheet
  --dielectric-method sheet2d
  --exciton-density 1e-5
  --probe-q-in-dq 2
  --eps-monitor
)