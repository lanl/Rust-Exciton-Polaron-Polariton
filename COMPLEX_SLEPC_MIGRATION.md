# Migration to Complex Hermitian Matrices in SLEPc

## Summary of Changes

The code has been updated to handle complex Hermitian matrices instead of real matrices. Here are the key changes:

## 1. Updated Files

### `light-matter/src/slepc.rs`
- Changed `diagonalize_slepc` to work with `SolverOperator<c64>` instead of `SolverOperator<f64>`
- Added `mv_product_complex` callback function that handles complex matrix-vector products
- Updated eigenvector extraction to handle complex vectors stored in PETSc's interleaved format
- Changed problem type from `NonHermitian` to `Hermitian` for better performance
- Updated `create_petsc_vecs_from_array2` to convert real initial guesses to complex format

### `slepc_rust/src/lib.rs`
- Added `create_complex_vector` function for creating complex PETSc vectors
- Added `get_complex_vector_array` function for extracting complex vector data
- Updated `get_vector_array` to be an alias for complex version

## 2. PETSc/SLEPc Complex Support

**CRITICAL**: PETSc and SLEPc must be compiled with complex scalar support!

### Quick Check

Run the provided script to check your installation:
```bash
chmod +x check_petsc_complex.sh
./check_petsc_complex.sh
```

### Manual Check

Check if your installation supports complex numbers:
```bash
grep "PETSC_USE_COMPLEX" $PETSC_DIR/$PETSC_ARCH/include/petscconf.h
```

If this returns nothing or shows `#define PETSC_USE_COMPLEX 0`, you need to recompile.

### Recompile PETSc with complex support:
```bash
cd $PETSC_DIR
./configure --with-scalar-type=complex \
            --with-precision=double \
            [add your other configuration options]
make all test

cd $SLEPC_DIR  
make all test
```

### Update your environment:
```bash
export PETSC_ARCH=arch-linux-c-complex  # or whatever you named it
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$SLEPC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
```

## 3. Data Format

### Complex Vector Storage in PETSc
PETSc stores complex vectors in **interleaved format**:
```
[real_0, imag_0, real_1, imag_1, ..., real_n, imag_n]
```

So a vector of `n` complex numbers requires an array of `2*n` doubles.

### Matrix-Vector Product
The `mv_product_complex` callback:
1. Receives PETSc vectors with interleaved real/imag data
2. Converts to `Vec<Complex64>` for Rust processing
3. Calls your operator's `apply` method
4. Converts result back to interleaved format

## 4. Eigenvector Output

Currently, the code returns **real parts only** of the eigenvectors:
```rust
eigenvectors[[j, i]] = re_part;  // Only storing real part
```

### If you need complex eigenvectors:

**Option A**: Return magnitude
```rust
eigenvectors[[j, i]] = (re_part * re_part + im_part * im_part).sqrt();
```

**Option B**: Change return type to complex
```rust
pub fn diagonalize_slepc<T: SolverOperator<c64>>(
    // ...
) -> (Array1<f64>, Array2<Complex64>) {  // Changed return type
    // ...
    let mut eigenvectors = Array2::zeros((op.dimension(), n_eig));
    // ...
    eigenvectors[[j, i]] = Complex64::new(re_part, im_part);
}
```

## 5. Hermitian vs Non-Hermitian

For **Hermitian matrices** (which yours should be):
- Eigenvalues are guaranteed to be real
- Eigenvectors can be chosen to be real (up to a phase)
- SLEPc can use more efficient algorithms
- The code now sets `EPSProblemType::Hermitian`

## 6. Testing

After recompiling PETSc/SLEPc with complex support:

```bash
# Rebuild your Rust project
cargo clean
cargo build --release

# Run the complex Hermitian tests
cargo test --test test_complex_hermitian -- --ignored --nocapture

# Run specific tests
cargo test --test test_complex_hermitian test_diagonal_hermitian -- --ignored --nocapture
cargo test --test test_complex_hermitian test_pauli_x -- --ignored --nocapture

# Test hermiticity of your operator
cargo test --test test_complex_hermitian test_hermiticity_check -- --ignored --nocapture

# Check that eigenvalues are real (for Hermitian matrices)
# Imaginary parts should be ~0 (within numerical precision)
```

### Using the Hermiticity Checker

You can verify your operator is Hermitian before diagonalization:

```rust
use light_matter::slepc::verify_hermitian;

// Before diagonalizing
if !verify_hermitian(&my_operator, 1e-10) {
    panic!("Operator is not Hermitian!");
}

// Then diagonalize
let (evals, evecs) = diagonalize_slepc(&my_operator, n_eig, block_size, solver_type, None);
```

## 7. Potential Issues

### Issue 1: Linking errors
If you get linking errors about undefined symbols, make sure:
- `PETSC_ARCH` points to your complex build
- `LD_LIBRARY_PATH` includes the complex library directories
- You've done `cargo clean` before rebuilding

### Issue 2: Wrong results
If eigenvalues/eigenvectors look wrong:
- Verify your operator's `apply` method works correctly with complex numbers
- Check that your Hamiltonian is actually Hermitian: `H = H†`
- Add debug prints to verify the matrix-vector product

### Issue 3: Performance
Complex arithmetic is ~4x slower than real arithmetic. If performance is critical:
- Ensure you're using optimized BLAS/LAPACK (MKL, OpenBLAS, etc.)
- Consider if your problem can be reformulated with real arithmetic
- Use appropriate solver types (Krylov-Schur is usually best for Hermitian)

## 8. Alternative: Real Arithmetic Formulation

If you can't or don't want to use complex PETSc, you can reformulate your problem:

For a complex Hermitian matrix `H`, you can create an equivalent real symmetric matrix:
```
H_real = [ Re(H)  -Im(H) ]
         [ Im(H)   Re(H) ]
```

This doubles the problem size but allows using real arithmetic. The eigenvalues appear in pairs.

## 9. Next Steps

1. ✅ Code updated to handle complex matrices
2. ⬜ Recompile PETSc/SLEPc with `--with-scalar-type=complex`
3. ⬜ Update environment variables
4. ⬜ Rebuild Rust project with `cargo clean && cargo build --release`
5. ⬜ Test with a small problem to verify correctness
6. ⬜ Decide if you need complex eigenvectors (update return type if needed)
7. ⬜ Run full calculations

## 10. Questions?

If you encounter issues:
- Check PETSc error messages carefully
- Verify matrix-vector product is correct with a small test case
- Ensure your Hamiltonian is Hermitian (eigenvalues should be real)
- Consider adding debug output to track data flow through the callbacks
