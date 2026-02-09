# Summary of Changes for Complex Hermitian Matrix Support

## Files Modified

### 1. `light-matter/src/slepc.rs`
**Purpose**: Main diagonalization interface using SLEPc

**Key Changes**:
- Changed `diagonalize_slepc` signature from `SolverOperator<f64>` to `SolverOperator<c64>`
- Added `mv_product_complex` callback function that:
  - Converts PETSc's interleaved complex format to Rust `Complex64`
  - Calls the operator's `apply` method
  - Converts back to interleaved format
- Updated eigenvector extraction to handle complex vectors
- Changed problem type from `NonHermitian` to `Hermitian` for better performance
- Updated `create_petsc_vecs_from_array2` to convert real initial guesses to complex format
- Added `verify_hermitian` function for debugging

**New Functions**:
```rust
fn mv_product_complex<T: SolverOperator<c64>>(...) -> c_int
pub fn verify_hermitian<T: SolverOperator<c64>>(op: &T, tolerance: f64) -> bool
```

### 2. `slepc_rust/src/lib.rs`
**Purpose**: Low-level Rust bindings to SLEPc/PETSc

**Key Changes**:
- Renamed `create_vector` to `create_complex_vector` with updated documentation
- Added `get_complex_vector_array` function for extracting complex vector data
- Made `get_vector_array` an alias for `get_complex_vector_array`

**New Functions**:
```rust
pub unsafe fn create_complex_vector(size: usize) -> Result<Vec, PetscError>
pub unsafe fn get_complex_vector_array(vec: Vec, array: &mut [f64]) -> Result<(), PetscError>
```

### 3. `light-matter/tests/test_complex_hermitian.rs` (NEW)
**Purpose**: Test suite for complex Hermitian matrix diagonalization

**Test Cases**:
- `test_diagonal_hermitian`: Tests diagonal matrix with known eigenvalues
- `test_pauli_x`: Tests Pauli X matrix (eigenvalues ±1)
- `test_hermiticity_check`: Verifies the hermiticity checker works
- `test_complex_matvec`: Unit test for complex matrix-vector product

### 4. `check_petsc_complex.sh` (NEW)
**Purpose**: Bash script to verify PETSc/SLEPc complex support

**Features**:
- Checks if PETSc is compiled with complex scalar support
- Verifies precision (double vs single)
- Checks for SLEPc library files
- Provides instructions for recompilation if needed

### 5. `COMPLEX_SLEPC_MIGRATION.md` (NEW)
**Purpose**: Comprehensive migration guide

**Contents**:
- Summary of all changes
- Instructions for recompiling PETSc/SLEPc
- Explanation of complex vector storage format
- Testing procedures
- Troubleshooting guide
- Alternative approaches

## Technical Details

### Complex Number Storage in PETSc

PETSc stores complex vectors in **interleaved format**:
```
[real_0, imag_0, real_1, imag_1, ..., real_n, imag_n]
```

A vector of `n` complex numbers requires an array of `2*n` doubles.

### Matrix-Vector Product Flow

1. **PETSc calls** `mv_product_complex` with vectors in interleaved format
2. **Callback converts** to `Vec<Complex64>`:
   ```rust
   for i in 0..n {
       x_complex[i] = c64::new(x_raw[2*i], x_raw[2*i + 1]);
   }
   ```
3. **Operator applies** transformation: `op.apply(&x_complex, &mut y_complex)`
4. **Callback converts back** to interleaved format:
   ```rust
   for i in 0..n {
       y_raw[2*i] = y_complex[i].re;
       y_raw[2*i + 1] = y_complex[i].im;
   }
   ```

### Hermitian Matrix Properties

For a Hermitian matrix H:
- **Eigenvalues are real**: λ ∈ ℝ
- **Eigenvectors can be chosen real** (up to a global phase)
- **Hermiticity condition**: H = H† (conjugate transpose)
- **Inner product property**: ⟨y, Hx⟩ = ⟨Hy, x⟩*

The `verify_hermitian` function checks this by testing:
```rust
⟨y, Hx⟩ ≈ ⟨Hy, x⟩*
```
for random vectors x and y.

## Requirements

### Compile-Time Requirements
- PETSc compiled with `--with-scalar-type=complex`
- SLEPc compiled against complex PETSc
- Rust dependencies: `num-complex`, `ndarray`, `rand`

### Runtime Requirements
- `PETSC_DIR` and `PETSC_ARCH` environment variables set
- `LD_LIBRARY_PATH` includes PETSc/SLEPc complex libraries
- Operator implements `SolverOperator<Complex64>`

## Migration Checklist

- [x] Update `slepc.rs` to handle complex numbers
- [x] Update `lib.rs` with complex vector functions
- [x] Create test suite for complex matrices
- [x] Create verification script
- [x] Write comprehensive documentation
- [ ] **Recompile PETSc with complex support** (user action required)
- [ ] **Recompile SLEPc** (user action required)
- [ ] **Update environment variables** (user action required)
- [ ] **Rebuild Rust project** (user action required)
- [ ] **Run tests to verify** (user action required)

## Usage Example

```rust
use light_matter::slepc::diagonalize_slepc;
use rust_slepc_bindings::{initialize, finalize, EPSType};

// Initialize SLEPc
initialize().expect("Failed to initialize SLEPc");

// Your operator must implement SolverOperator<Complex64>
let my_operator = MyHermitianOperator::new(...);

// Optional: verify hermiticity
use light_matter::slepc::verify_hermitian;
assert!(verify_hermitian(&my_operator, 1e-10));

// Diagonalize
let (eigenvalues, eigenvectors) = diagonalize_slepc(
    &my_operator,
    n_eig,           // number of eigenvalues
    block_size,      // block size for block methods
    Some(EPSType::KrylovSchur),  // solver type
    None,            // optional initial guess
);

// eigenvalues: Array1<f64> - real eigenvalues
// eigenvectors: Array2<f64> - real parts of eigenvectors

// Finalize SLEPc
finalize().expect("Failed to finalize SLEPc");
```

## Performance Considerations

1. **Complex arithmetic is ~4x slower** than real arithmetic
2. **Memory usage doubles** (real + imaginary parts)
3. **Use optimized BLAS/LAPACK** (MKL, OpenBLAS) for best performance
4. **Hermitian solvers are faster** than general non-Hermitian solvers
5. **Block methods** (GD, JD) can be more efficient for multiple eigenvalues

## Troubleshooting

### Linking Errors
```
undefined reference to `VecCreate_Seq_Complex`
```
**Solution**: PETSc not compiled with complex support. Run `./check_petsc_complex.sh`

### Wrong Results
- Verify operator is Hermitian: `verify_hermitian(&op, 1e-10)`
- Check eigenvalues are real (imaginary parts should be ~0)
- Add debug prints in `mv_product_complex` to verify data flow

### Performance Issues
- Use `EPSType::KrylovSchur` for Hermitian matrices
- Increase block size for block methods
- Ensure using optimized BLAS/LAPACK
- Consider if problem can be reformulated with real arithmetic

## Next Steps

1. Run `./check_petsc_complex.sh` to verify your PETSc installation
2. If needed, recompile PETSc/SLEPc with complex support
3. Run `cargo test --test test_complex_hermitian -- --ignored --nocapture`
4. Verify your operator with `verify_hermitian`
5. Proceed with your calculations

## Questions or Issues?

- Check `COMPLEX_SLEPC_MIGRATION.md` for detailed instructions
- Review test cases in `test_complex_hermitian.rs` for examples
- Verify PETSc configuration with `./check_petsc_complex.sh`
- Add debug output to track data flow through callbacks
