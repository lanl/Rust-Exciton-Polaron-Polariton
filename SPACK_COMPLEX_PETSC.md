# Installing Complex PETSc/SLEPc with Spack

## Your Current Situation

You have PETSc installed via Spack (indicated by the hash in the directory name).
Spack-managed installations don't have a `./configure` script - they're pre-built.

## Solution: Install Complex PETSc via Spack

### Step 1: Check Current Installation

```bash
# See what PETSc variants you have
spack find -lv petsc

# Check if you have complex PETSc
spack find petsc +complex
```

### Step 2: Install Complex PETSc

```bash
# Install PETSc with complex scalar support
spack install petsc@3.21.1 +complex +double ~int64

# Or with more options (recommended):
spack install petsc@3.21.1 +complex +double ~int64 +mpi +hdf5
```

### Step 3: Install Complex SLEPc

```bash
# Install SLEPc that depends on complex PETSc
spack install slepc@3.21.1 ^petsc+complex
```

### Step 4: Load the Complex Versions

```bash
# Unload current PETSc/SLEPc
spack unload petsc slepc

# Load complex versions
spack load petsc+complex
spack load slepc ^petsc+complex

# Verify
echo $PETSC_DIR
echo $SLEPC_DIR
```

### Step 5: Update Your Environment

Add to your `~/.bashrc` or `~/.bash_profile`:

```bash
# Load complex PETSc/SLEPc
spack load petsc+complex
spack load slepc ^petsc+complex

# These should be set automatically by spack load
export PETSC_DIR=$(spack location -i petsc+complex)
export SLEPC_DIR=$(spack location -i slepc ^petsc+complex)
export PETSC_ARCH=""  # Spack doesn't use PETSC_ARCH

# Add to library path
export LD_LIBRARY_PATH=$PETSC_DIR/lib:$SLEPC_DIR/lib:$LD_LIBRARY_PATH
```

### Step 6: Verify Installation

```bash
# Check PETSc configuration
grep "PETSC_USE_COMPLEX" $PETSC_DIR/include/petscconf.h

# Should show:
# #define PETSC_USE_COMPLEX 1

# Or use the check script
cd /path/to/your/rust/project
./check_petsc_complex.sh
```

### Step 7: Rebuild Your Rust Project

```bash
cd /path/to/your/rust/project
cargo clean
cargo build --release
```

## Alternative: Use Existing Complex PETSc (if available)

If your system already has complex PETSc installed:

```bash
# Search for it
spack find -lv petsc | grep complex

# If found, load it
spack load /hash-of-complex-petsc
spack load slepc ^/hash-of-complex-petsc
```

## Spack Spec Details

### Minimal Complex PETSc
```bash
spack install petsc@3.21.1 +complex +double
```

### Recommended Complex PETSc
```bash
spack install petsc@3.21.1 \
    +complex \
    +double \
    ~int64 \
    +mpi \
    +hdf5 \
    +hypre \
    +metis \
    +mumps \
    +superlu-dist
```

### Key Variants
- `+complex`: Enable complex scalar support (REQUIRED)
- `+double`: Use double precision (REQUIRED)
- `~int64`: Use 32-bit integers (usually fine)
- `+mpi`: Enable MPI support
- `+hdf5`: Enable HDF5 for I/O
- `+hypre`, `+metis`, `+mumps`, `+superlu-dist`: Optional solvers/preconditioners

## Checking Your Installation

### Method 1: Using Spack
```bash
spack find -lv petsc
# Look for "+complex" in the output
```

### Method 2: Check Header File
```bash
PETSC_DIR=$(spack location -i petsc+complex)
grep "PETSC_USE_COMPLEX" $PETSC_DIR/include/petscconf.h
```

### Method 3: Use Our Script
```bash
# Update the script to work with Spack
export PETSC_DIR=$(spack location -i petsc+complex)
export PETSC_ARCH=""
./check_petsc_complex.sh
```

## Troubleshooting

### Issue: `spack install` fails

**Try with fewer dependencies:**
```bash
spack install petsc@3.21.1 +complex +double ~int64 ~mpi
```

### Issue: Multiple PETSc versions conflict

**Use a Spack environment:**
```bash
# Create environment
spack env create complex-petsc
spack env activate complex-petsc

# Install in environment
spack install petsc+complex
spack install slepc ^petsc+complex

# Use environment
spack env activate complex-petsc
```

### Issue: Can't find complex PETSc after install

**Get the installation path:**
```bash
spack location -i petsc+complex
```

**Load explicitly:**
```bash
spack load $(spack find --format '{hash}' petsc+complex)
```

## Quick Setup Script

Create a file `setup_complex_petsc.sh`:

```bash
#!/bin/bash

# Install complex PETSc/SLEPc if not already installed
if ! spack find petsc+complex &> /dev/null; then
    echo "Installing complex PETSc..."
    spack install petsc@3.21.1 +complex +double ~int64
fi

if ! spack find slepc ^petsc+complex &> /dev/null; then
    echo "Installing SLEPc with complex PETSc..."
    spack install slepc@3.21.1 ^petsc+complex
fi

# Load the packages
echo "Loading complex PETSc/SLEPc..."
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

# Verify
if grep -q "define PETSC_USE_COMPLEX 1" $PETSC_DIR/include/petscconf.h; then
    echo "✓ Complex PETSc is properly configured"
else
    echo "✗ WARNING: PETSc may not have complex support"
fi
```

Then use it:
```bash
chmod +x setup_complex_petsc.sh
source setup_complex_petsc.sh
```

## For Your Specific Case

Based on your directory name, you're using Spack. Here's what to do:

```bash
# 1. Check if complex PETSc is already installed
spack find -lv petsc | grep complex

# 2. If not, install it
spack install petsc@3.21.1 +complex +double

# 3. Install matching SLEPc
spack install slepc@3.21.1 ^petsc+complex

# 4. Load them
spack unload petsc slepc  # Unload current versions
spack load petsc+complex
spack load slepc ^petsc+complex

# 5. Verify
export PETSC_DIR=$(spack location -i petsc+complex)
grep "PETSC_USE_COMPLEX" $PETSC_DIR/include/petscconf.h

# 6. Rebuild Rust
cd /path/to/rust/project
cargo clean
cargo build --release
```

## Adding to Your Shell Profile

Add to `~/.bashrc`:

```bash
# Load complex PETSc/SLEPc for Rust project
if command -v spack &> /dev/null; then
    spack load petsc+complex 2>/dev/null
    spack load slepc ^petsc+complex 2>/dev/null
    
    export PETSC_DIR=$(spack location -i petsc+complex 2>/dev/null)
    export SLEPC_DIR=$(spack location -i slepc ^petsc+complex 2>/dev/null)
    export PETSC_ARCH=""
    
    if [ -n "$PETSC_DIR" ]; then
        export LD_LIBRARY_PATH=$PETSC_DIR/lib:$SLEPC_DIR/lib:$LD_LIBRARY_PATH
    fi
fi
```

Then:
```bash
source ~/.bashrc
```
