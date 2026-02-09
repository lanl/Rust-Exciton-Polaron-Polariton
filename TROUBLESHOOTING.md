# Troubleshooting Guide: Complex Hermitian Diagonalization

## Table of Contents
1. [Compilation Errors](#compilation-errors)
2. [Linking Errors](#linking-errors)
3. [Runtime Errors](#runtime-errors)
4. [Incorrect Results](#incorrect-results)
5. [Performance Issues](#performance-issues)
6. [Memory Issues](#memory-issues)

---

## Compilation Errors

### Error: `cannot find type 'c64' in this scope`

**Cause**: Missing import for Complex64

**Solution**:
```rust
use num_complex::Complex64 as c64;
```

### Error: `trait bound 'MyOperator: SolverOperator<f64>' is not satisfied`

**Cause**: Operator still implements `SolverOperator<f64>` instead of `SolverOperator<c64>`

**Solution**: Update your operator implementation:
```rust
// Old:
impl SolverOperator<f64> for MyOperator { ... }

// New:
impl SolverOperator<c64> for MyOperator { ... }
```

---

## Linking Errors

### Error: `undefined reference to 'VecCreate_Seq_Complex'`

**Cause**: PETSc not compiled with complex scalar support

**Solution**:
```bash
# Check PETSc configuration
./check_petsc_complex.sh

# If not complex, recompile:
cd $PETSC_DIR
./configure --with-scalar-type=complex --with-precision=double
make all test

cd $SLEPC_DIR
make all test

# Update environment
export PETSC_ARCH=arch-linux-c-complex
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$SLEPC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH

# Rebuild Rust
cargo clean
cargo build --release
```

---

## Runtime Errors

### Error: `Segmentation fault` in `mv_product_complex`

**Cause**: Invalid pointer access or dimension mismatch

**Solution**: Add bounds checking and verify dimensions match

---

## Incorrect Results

### Problem: Eigenvalues have large imaginary parts

**Cause**: Matrix is not Hermitian

**Solution**: Verify hermiticity:
```rust
use light_matter::slepc::verify_hermitian;

if !verify_hermitian(&op, 1e-10) {
    panic!("Operator is not Hermitian!");
}
```

---

## Performance Issues

### Problem: Diagonalization is very slow

**Solutions**:
1. Profile your `apply()` method
2. Use appropriate solver (KrylovSchur for Hermitian)
3. Optimize BLAS/LAPACK
4. Reduce problem size

---

## Debugging Checklist

- [ ] PETSc compiled with `--with-scalar-type=complex`
- [ ] Environment variables set correctly
- [ ] Operator implements `SolverOperator<Complex64>`
- [ ] Matrix is Hermitian
- [ ] Dimensions are consistent
- [ ] Using appropriate solver type

## Getting Help

1. Run `./check_petsc_complex.sh`
2. Run tests: `cargo test --test test_complex_hermitian -- --ignored`
3. Check `DATA_FLOW_DIAGRAM.md`
4. Review `COMPLEX_SLEPC_MIGRATION.md`
