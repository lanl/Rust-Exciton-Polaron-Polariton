#!/bin/bash
# Check if PETSc/SLEPc are compiled with complex scalar support
# Works with both source installations and Spack installations

echo "Checking PETSc/SLEPc complex scalar support..."
echo ""

# Check if using Spack
if command -v spack &> /dev/null; then
    echo "Detected Spack installation"
    
    # Try to find complex PETSc
    if spack find petsc+complex &> /dev/null; then
        echo "✓ Found complex PETSc in Spack"
        
        # Try to load it
        spack load petsc+complex 2>/dev/null
        
        # Get location
        PETSC_DIR=$(spack location -i petsc+complex 2>/dev/null)
        if [ -n "$PETSC_DIR" ]; then
            echo "  Location: $PETSC_DIR"
            export PETSC_DIR
            export PETSC_ARCH=""
        fi
    else
        echo "✗ Complex PETSc not found in Spack"
        echo ""
        echo "To install:"
        echo "  spack install petsc@3.21.1 +complex +double"
        echo "  spack install slepc@3.21.1 ^petsc+complex"
        echo ""
        echo "See SPACK_COMPLEX_PETSC.md for detailed instructions"
        exit 1
    fi
    echo ""
fi

# Check environment variables
if [ -z "$PETSC_DIR" ]; then
    echo "ERROR: PETSC_DIR is not set"
    
    # Try to find it with Spack
    if command -v spack &> /dev/null; then
        echo "Attempting to locate with Spack..."
        PETSC_DIR=$(spack location -i petsc+complex 2>/dev/null)
        if [ -n "$PETSC_DIR" ]; then
            echo "Found: $PETSC_DIR"
            export PETSC_DIR
            export PETSC_ARCH=""
        else
            echo "Run: spack load petsc+complex"
            exit 1
        fi
    else
        exit 1
    fi
fi

if [ -z "$PETSC_ARCH" ]; then
    echo "INFO: PETSC_ARCH is not set (OK for Spack installations)"
    PETSC_ARCH=""
fi

echo "PETSC_DIR: $PETSC_DIR"
echo "PETSC_ARCH: $PETSC_ARCH"
echo ""

# Check petscconf.h
PETSC_CONF="$PETSC_DIR/$PETSC_ARCH/include/petscconf.h"

# For Spack installations, try without PETSC_ARCH
if [ ! -f "$PETSC_CONF" ] && [ -z "$PETSC_ARCH" ]; then
    PETSC_CONF="$PETSC_DIR/include/petscconf.h"
fi

if [ ! -f "$PETSC_CONF" ]; then
    echo "ERROR: Cannot find $PETSC_CONF"
    echo "Make sure PETSC_DIR and PETSC_ARCH are set correctly"
    exit 1
fi

echo "Checking $PETSC_CONF..."

# Check for complex support
if grep -q "define PETSC_USE_COMPLEX 1" "$PETSC_CONF"; then
    echo "✓ PETSc is compiled with COMPLEX scalar support"
    COMPLEX_SUPPORT=1
elif grep -q "define PETSC_USE_COMPLEX 0" "$PETSC_CONF"; then
    echo "✗ PETSc is compiled with REAL scalar support"
    echo ""
    echo "You need to recompile PETSc with complex support:"
    echo "  cd \$PETSC_DIR"
    echo "  ./configure --with-scalar-type=complex [other options]"
    echo "  make all test"
    COMPLEX_SUPPORT=0
else
    echo "? Cannot determine PETSc scalar type"
    COMPLEX_SUPPORT=0
fi

echo ""

# Check precision
if grep -q "define PETSC_USE_REAL_DOUBLE 1" "$PETSC_CONF"; then
    echo "✓ PETSc is using DOUBLE precision"
elif grep -q "define PETSC_USE_REAL_SINGLE 1" "$PETSC_CONF"; then
    echo "⚠ PETSc is using SINGLE precision (may cause accuracy issues)"
else
    echo "? Cannot determine PETSc precision"
fi

echo ""

# Check SLEPc
if [ -z "$SLEPC_DIR" ]; then
    echo "WARNING: SLEPC_DIR is not set"
    
    # Try to find it with Spack
    if command -v spack &> /dev/null; then
        echo "Attempting to locate with Spack..."
        SLEPC_DIR=$(spack location -i slepc ^petsc+complex 2>/dev/null)
        if [ -n "$SLEPC_DIR" ]; then
            echo "Found: $SLEPC_DIR"
            export SLEPC_DIR
        fi
    fi
else
    echo "SLEPC_DIR: $SLEPC_DIR"
    
    # Check for library (try both with and without PETSC_ARCH)
    SLEPC_LIB="$SLEPC_DIR/$PETSC_ARCH/lib/libslepc.so"
    if [ ! -f "$SLEPC_LIB" ]; then
        SLEPC_LIB="$SLEPC_DIR/lib/libslepc.so"
    fi
    
    if [ -f "$SLEPC_LIB" ]; then
        echo "✓ Found SLEPc library: $SLEPC_LIB"
    else
        SLEPC_LIB="$SLEPC_DIR/$PETSC_ARCH/lib/libslepc.a"
        if [ ! -f "$SLEPC_LIB" ]; then
            SLEPC_LIB="$SLEPC_DIR/lib/libslepc.a"
        fi
        
        if [ -f "$SLEPC_LIB" ]; then
            echo "✓ Found SLEPc library: $SLEPC_LIB"
        else
            echo "✗ Cannot find SLEPc library"
        fi
    fi
fi

echo ""
echo "Summary:"
if [ $COMPLEX_SUPPORT -eq 1 ]; then
    echo "✓ Your PETSc/SLEPc installation supports complex scalars"
    echo "  You can proceed with building the Rust code"
    echo ""
    echo "Next steps:"
    echo "  1. cargo clean"
    echo "  2. cargo build --release"
    echo "  3. cargo test --test test_complex_hermitian -- --ignored --nocapture"
    exit 0
else
    echo "✗ Your PETSc/SLEPc installation does NOT support complex scalars"
    echo ""
    
    if command -v spack &> /dev/null; then
        echo "You are using Spack. To install complex PETSc:"
        echo "  1. Install complex PETSc:"
        echo "     spack install petsc@3.21.1 +complex +double"
        echo ""
        echo "  2. Install matching SLEPc:"
        echo "     spack install slepc@3.21.1 ^petsc+complex"
        echo ""
        echo "  3. Load the packages:"
        echo "     spack load petsc+complex"
        echo "     spack load slepc ^petsc+complex"
        echo ""
        echo "  4. Rebuild Rust code:"
        echo "     cargo clean"
        echo "     cargo build --release"
        echo ""
        echo "See SPACK_COMPLEX_PETSC.md for detailed instructions"
    else
        echo "To fix this:"
        echo "  1. Reconfigure PETSc with complex support:"
        echo "     cd \$PETSC_DIR"
        echo "     ./configure --with-scalar-type=complex --with-precision=double [other options]"
        echo "     make all test"
        echo ""
        echo "  2. Rebuild SLEPc:"
        echo "     cd \$SLEPC_DIR"
        echo "     make all test"
        echo ""
        echo "  3. Update your environment (add to ~/.bashrc):"
        echo "     export PETSC_ARCH=arch-linux-c-complex  # or whatever you named it"
        echo "     export LD_LIBRARY_PATH=\$PETSC_DIR/\$PETSC_ARCH/lib:\$SLEPC_DIR/\$PETSC_ARCH/lib:\$LD_LIBRARY_PATH"
        echo ""
        echo "  4. Rebuild Rust code:"
        echo "     cargo clean"
        echo "     cargo build --release"
    fi
    exit 1
fi
