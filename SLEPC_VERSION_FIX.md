# Fixing SLEPc Installation Issue

## The Problem

SLEPc 3.21.1 is not available in your Spack repository.

## Solution: Install Available SLEPc Version

```bash
# 1. Check what SLEPc versions are available
spack info slepc

# 2. Install the latest available version that matches PETSc 3.21
spack install slepc@3.21 ^petsc+complex

# Or try without specifying exact version (gets latest compatible)
spack install slepc ^petsc+complex
```

## If That Doesn't Work

Try these alternatives:

### Option 1: Use Latest Available SLEPc
```bash
# Find available versions
spack versions slepc

# Install latest (e.g., if 3.20 is available)
spack install slepc@3.20 ^petsc+complex
```

### Option 2: Let Spack Choose Compatible Version
```bash
# This will install whatever SLEPc version is compatible with your PETSc
spack install slepc ^petsc+complex
```

### Option 3: Update Spack Repository
```bash
# Update Spack to get newer package definitions
cd ~/spack
git pull
spack install slepc@3.21 ^petsc+complex
```

## After Installation

Once SLEPc is installed (regardless of exact version):

```bash
# 1. Load the packages
spack load petsc+complex
spack load slepc ^petsc+complex

# 2. Verify
export PETSC_DIR=$(spack location -i petsc+complex)
export SLEPC_DIR=$(spack location -i slepc ^petsc+complex)

grep "PETSC_USE_COMPLEX" $PETSC_DIR/include/petscconf.h
# Should show: #define PETSC_USE_COMPLEX 1

# 3. Check versions
ls $PETSC_DIR/include/petscversion.h
ls $SLEPC_DIR/include/slepcversion.h

# 4. Rebuild Rust
cd ~/two_e_polaron-polariton/rust_sub
cargo clean
cargo build --release
```

## Checking What You Have

```bash
# See what PETSc you just installed
spack find -lv petsc+complex

# See what SLEPc versions are available
spack versions slepc

# See all installed packages
spack find
```

## Quick Commands for Your Situation

```bash
# Try these in order until one works:

# 1. Try without exact version
spack install slepc ^petsc+complex

# 2. If that fails, try 3.21 (without .1)
spack install slepc@3.21 ^petsc+complex

# 3. If that fails, try 3.20
spack install slepc@3.20 ^petsc+complex

# 4. If that fails, see what's available
spack versions slepc
# Then install the latest version shown
```

## After Successful Installation

```bash
# Load packages
spack load petsc+complex
spack load slepc ^petsc+complex

# Verify
cd ~/two_e_polaron-polariton/rust_sub
./check_petsc_complex.sh

# Should show:
# ✓ PETSc is compiled with COMPLEX scalar support
# ✓ Found SLEPc library

# Build Rust
cargo clean
cargo build --release

# Test
cargo test --test test_complex_hermitian -- --ignored --nocapture
```

## Version Compatibility

SLEPc and PETSc versions should match:
- PETSc 3.21.x → SLEPc 3.21.x
- PETSc 3.20.x → SLEPc 3.20.x

But minor version differences are usually OK:
- PETSc 3.21.1 + SLEPc 3.21.0 ✓ (should work)
- PETSc 3.21.1 + SLEPc 3.20.x ⚠️ (might work)
- PETSc 3.21.1 + SLEPc 3.19.x ✗ (likely won't work)
