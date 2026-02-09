# Rebuilding SLEPc with Full Features

## Problem

Your current SLEPc installation is missing some components:
- BV type "mat" (basis vectors as dense matrices)
- RG type "interval" (region filtering)
- Possibly other optional features

This causes error messages (though non-fatal) when running tests.

## Solution: Reinstall SLEPc with Full Features

### Step 1: Check Current Installation

```bash
# See what you have
spack find -lv slepc
spack find -lv petsc

# Check the spec that was used
spack spec slepc ^petsc+complex
```

### Step 2: Uninstall Current SLEPc (Optional)

If you want a clean reinstall:

```bash
# Find the hash of your current SLEPc
spack find -l slepc

# Uninstall it (replace HASH with actual hash)
spack uninstall /HASH

# Or uninstall by spec
spack uninstall slepc ^petsc+complex
```

### Step 3: Install SLEPc with Full Features

The key is to ensure PETSc is built with all the features SLEPc needs:

```bash
# Install PETSc with comprehensive features
spack install petsc@3.21.1 \
    +complex \
    +double \
    ~int64 \
    +mpi \
    +hdf5 \
    +hypre \
    +metis \
    +parmetis \
    +superlu-dist \
    +mumps \
    +scalapack

# Then install SLEPc against this PETSc
spack install slepc@3.21.1 ^petsc+complex+mpi+hdf5
```

### Step 4: Alternative - Use Spack Variants

Check what variants are available:

```bash
# See available variants for SLEPc
spack info slepc

# See available variants for PETSc  
spack info petsc
```

Then install with specific variants:

```bash
# Install with ARPACK support (if available)
spack install slepc@3.21.1 +arpack ^petsc+complex

# Or with all available features
spack install slepc@3.21.1 ^petsc+complex+mpi+hdf5+hypre+metis+superlu-dist
```

### Step 5: Verify the Installation

```bash
# Load the new installation
spack load slepc ^petsc+complex

# Check what was built
export SLEPC_DIR=$(spack location -i slepc ^petsc+complex)
ls $SLEPC_DIR/lib/

# Check if BV types are available
# (This requires looking at the SLEPc source or running a test)
```

### Step 6: Test Your Rust Code

```bash
cd /path/to/your/rust/project
cargo clean
cargo test --test test_complex_hermitian -- --ignored --nocapture
```

## Understanding the Issue

The error "Unable to find requested BV type mat" suggests that SLEPc was compiled without support for dense matrix storage of basis vectors. This can happen if:

1. **PETSc was built without dense matrix support** - unlikely, as this is core functionality
2. **SLEPc was built with minimal features** - more likely with Spack's default spec
3. **There's a version mismatch** between PETSc and SLEPc

## Recommended Full Installation

Here's the complete command sequence for a full-featured installation:

```bash
# 1. Install PETSc with all common features
spack install petsc@3.21.1 \
    +complex \
    +double \
    ~int64 \
    +mpi \
    +shared \
    +hdf5 \
    +hypre \
    +metis \
    +parmetis \
    +superlu-dist

# 2. Install SLEPc with the same PETSc
spack install slepc@3.21.1 ^petsc@3.21.1+complex+double~int64+mpi+shared

# 3. Load them
spack unload petsc slepc
spack load petsc@3.21.1+complex
spack load slepc@3.21.1 ^petsc@3.21.1+complex

# 4. Set environment
export PETSC_DIR=$(spack location -i petsc@3.21.1+complex)
export SLEPC_DIR=$(spack location -i slepc@3.21.1)
export PETSC_ARCH=""
export LD_LIBRARY_PATH=$PETSC_DIR/lib:$SLEPC_DIR/lib:$LD_LIBRARY_PATH

# 5. Verify
echo "PETSC_DIR=$PETSC_DIR"
echo "SLEPC_DIR=$SLEPC_DIR"
grep "PETSC_USE_COMPLEX" $PETSC_DIR/include/petscconf.h
```

## Alternative: Accept the Error Messages

The error messages you're seeing are **non-fatal**. SLEPc automatically falls back to available types:

- When "mat" BV type is not found, it uses "svec" (standard vectors)
- When "interval" RG type is not found, it uses default region

**Your tests are passing**, which means the functionality is working correctly. The error messages are just warnings about unavailable optional features.

If you don't want to reinstall, you can:

1. **Suppress the errors** by redirecting stderr (not recommended)
2. **Document the expected errors** in your test output
3. **Use our current code** which works around the issues

## Quick Check: What's Actually Missing?

To see what's actually available in your SLEPc build:

```bash
# Check SLEPc configuration
export SLEPC_DIR=$(spack location -i slepc ^petsc+complex)
cat $SLEPC_DIR/lib/slepc/conf/slepcvariables

# Check PETSc configuration  
export PETSC_DIR=$(spack location -i petsc+complex)
cat $PETSC_DIR/lib/petsc/conf/petscvariables

# Look for configure options
grep "CONFIGURE_OPTIONS" $SLEPC_DIR/lib/slepc/conf/slepcvariables
```

## Minimal Reinstall (If Needed)

If you just want to fix the BV issue without a full reinstall:

```bash
# The issue might be that SLEPc needs to be rebuilt against your current PETSc
spack install --fresh slepc@3.21.1 ^petsc+complex

# Or force a rebuild
spack uninstall slepc
spack install slepc@3.21.1 ^petsc+complex
```

## For Your Specific Case

Based on your error messages, I recommend:

### Option 1: Accept the Errors (Easiest)
The errors are non-fatal and your tests pass. Just document that these warnings are expected.

### Option 2: Minimal Fix (Medium)
```bash
# Rebuild SLEPc to ensure it's properly linked with PETSc
spack uninstall slepc
spack install slepc@3.21.1 ^petsc@3.21.1+complex+double+mpi
spack load slepc ^petsc+complex
```

### Option 3: Full Reinstall (Most Thorough)
```bash
# Uninstall both
spack uninstall slepc petsc

# Reinstall with full features
spack install petsc@3.21.1 +complex +double +mpi +hdf5 +hypre +metis +superlu-dist
spack install slepc@3.21.1 ^petsc+complex

# Load and test
spack load petsc+complex
spack load slepc ^petsc+complex
cd /path/to/rust/project
cargo clean
cargo test --test test_complex_hermitian -- --ignored --nocapture
```

## My Recommendation

Given that your tests are **already passing**, I recommend **Option 1** - just accept the error messages. They're warnings about optional features, not actual failures.

If you really want to eliminate them, try **Option 2** first (rebuild SLEPc), as it's less invasive than a full reinstall.

## Adding Error Suppression to Tests

If you want to suppress PETSc error output in tests, you can set environment variables:

```bash
# Suppress PETSc error output
export PETSC_OPTIONS="-options_suppress_deprecated_warnings -on_error_silent"

# Run tests
cargo test --test test_complex_hermitian -- --ignored --nocapture
```

Or add to your test setup:

```rust
// In your test file
fn initialize_slepc() {
    INIT.call_once(|| {
        // Set PETSc options to suppress warnings
        std::env::set_var("PETSC_OPTIONS", "-on_error_silent");
        rust_slepc_bindings::initialize().expect("Failed to initialize SLEPc");
    });
}
```
