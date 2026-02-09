# Complex PETSc/SLEPc Bindings Fix

## Problem

When PETSc/SLEPc is compiled with `--with-scalar-type=complex`, the scalar type `PetscScalar` is a complex number (typically `double _Complex` in C, which is 16 bytes = 2 doubles).

The original Rust bindings were written for **real** PETSc, where:
- `EPSGetEigenvalue(eps, i, &kr, &ki)` expects `kr` and `ki` to be `double*` (8 bytes each)
- The eigenvalue is returned as separate real (`kr`) and imaginary (`ki`) parts

However, for **complex** PETSc:
- `EPSGetEigenvalue(eps, i, &kr, &ki)` expects `kr` and `ki` to be `PetscScalar*` (16 bytes each)
- Each `PetscScalar` is a complex number with real and imaginary parts
- For Hermitian matrices, the eigenvalue is real, so it's stored in `kr` with `kr.imag = 0`

## The Segfault

The segfault occurred at line 234 of `slepc.rs`:
```rust
let re = solver.get_eigenvalue(i).expect("Failed to get eigenvalue");
```

This was because:
1. We were passing `&mut f64` (8 bytes) to a function expecting `&mut PetscScalar` (16 bytes)
2. PETSc tried to write 16 bytes into an 8-byte location, causing memory corruption
3. This led to a segmentation fault

## The Fix

### In `slepc_rust/src/lib.rs`

Changed the FFI declarations from:
```rust
pub fn EPSGetEigenvalue(eps: EPS, i: c_int, kr: *mut f64, ki: *mut f64) -> PetscErrorCode;
```

To:
```rust
pub fn EPSGetEigenvalue(eps: EPS, i: c_int, kr: *mut [f64; 2], ki: *mut [f64; 2]) -> PetscErrorCode;
```

This correctly represents that each parameter is a pointer to a complex scalar (2 doubles).

### In `Eigensolver::get_eigenvalue()`

Changed from:
```rust
let mut kr: f64 = 0.0;
let mut ki: f64 = 0.0;
let result = unsafe { EPSGetEigenvalue(self.eps, i as c_int, &mut kr, &mut ki) };
```

To:
```rust
let mut eigval = [0.0_f64, 0.0_f64];  // [real, imag]
let mut dummy = [0.0_f64, 0.0_f64];   // unused for standard problems
let result = unsafe { EPSGetEigenvalue(self.eps, i as c_int, &mut eigval, &mut dummy) };
if result == PETSC_SUCCESS { Ok(eigval[0]) } else { Err(PetscError(result)) }
```

This allocates the correct amount of space (16 bytes per parameter) and extracts the real part.

### In `Eigensolver::get_eigenpair()`

Changed to use `EPSGetEigenvector` instead of `EPSGetEigenpair` to avoid dealing with eigenvalue extraction in this function. The eigenvalue can be retrieved separately with `get_eigenvalue()`.

## Memory Layout

For complex PETSc, a `PetscScalar` is laid out in memory as:
```
[real_part: f64, imaginary_part: f64]
```

So when we declare `let mut eigval = [0.0_f64, 0.0_f64]`, we're creating:
- `eigval[0]` = real part
- `eigval[1]` = imaginary part

For Hermitian matrices:
- Eigenvalues are real, so `eigval[1]` ≈ 0
- We return `eigval[0]` as the eigenvalue

## Testing

After this fix, the code should:
1. Not segfault when calling `get_eigenvalue()`
2. Correctly extract real eigenvalues from Hermitian matrices
3. Work with complex eigenvectors (already handled correctly in the vector operations)

## Related Files

- `slepc_rust/src/lib.rs` - FFI bindings and `Eigensolver` implementation
- `light-matter/src/slepc.rs` - High-level diagonalization function using SLEPc

## References

- PETSc documentation on complex numbers: https://petsc.org/release/manual/vec/#sec-complex
- SLEPc manual on eigenvalue extraction
- PETSc scalar types: When compiled with `--with-scalar-type=complex`, `PetscScalar` is `double _Complex`
