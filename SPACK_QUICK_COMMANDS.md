# Quick Commands for Your Spack Setup

## Immediate Next Steps

```bash
# 1. Check if complex PETSc is already installed
spack find -lv petsc | grep complex

# 2. If not found, install it (this may take 10-30 minutes)
spack install petsc@3.21.1 +complex +double

# 3. Install matching SLEPc
spack install slepc@3.21.1 ^petsc+complex

# 4. Load the packages
spack load petsc+complex
spack load slepc ^petsc+complex

# 5. Verify installation
export PETSC_DIR=$(spack location -i petsc+complex)
grep "PETSC_USE_COMPLEX" $PETSC_DIR/include/petscconf.h
# Should show: #define PETSC_USE_COMPLEX 1

# 6. Go to your Rust project and rebuild
cd ~/two_e_polaron-polariton/rust_sub
cargo clean
cargo build --release

# 7. Run tests
cargo test --test test_complex_hermitian -- --ignored --nocapture
```

## Add to Your ~/.bashrc

Add these lines to automatically load complex PETSc when you log in:

```bash
# Add to ~/.bashrc
if command -v spack &> /dev/null; then
    spack load petsc+complex 2>/dev/null
    spack load slepc ^petsc+complex 2>/dev/null
fi
```

Then reload:
```bash
source ~/.bashrc
```

## Useful Spack Commands

```bash
# See all installed PETSc versions
spack find -lv petsc

# See detailed info about complex PETSc
spack info petsc

# See what's loaded
spack find --loaded

# Unload current PETSc/SLEPc
spack unload petsc slepc

# Load complex versions
spack load petsc+complex
spack load slepc ^petsc+complex

# Get installation directory
spack location -i petsc+complex
spack location -i slepc ^petsc+complex
```

## If Installation Fails

Try with minimal dependencies:
```bash
spack install petsc@3.21.1 +complex +double ~int64 ~mpi
spack install slepc@3.21.1 ^petsc+complex
```

## Checking Your Setup

```bash
# Run the check script
cd ~/two_e_polaron-polariton/rust_sub
./check_petsc_complex.sh

# Should show:
# ✓ PETSc is compiled with COMPLEX scalar support
```

## Environment Variables

After loading with Spack, these should be set:

```bash
echo $PETSC_DIR    # Should show path to complex PETSc
echo $SLEPC_DIR    # Should show path to SLEPc
echo $PETSC_ARCH   # Should be empty for Spack (that's OK)
```

## Full Setup Script

Save this as `setup_env.sh`:

```bash
#!/bin/bash

# Load complex PETSc/SLEPc
spack load petsc+complex
spack load slepc ^petsc+complex

# Set environment variables
export PETSC_DIR=$(spack location -i petsc+complex)
export SLEPC_DIR=$(spack location -i slepc ^petsc+complex)
export PETSC_ARCH=""
export LD_LIBRARY_PATH=$PETSC_DIR/lib:$SLEPC_DIR/lib:$LD_LIBRARY_PATH

echo "Environment configured:"
echo "  PETSC_DIR=$PETSC_DIR"
echo "  SLEPC_DIR=$SLEPC_DIR"
```

Then use it:
```bash
source setup_env.sh
```
