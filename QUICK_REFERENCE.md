# Quick Reference: Complex Hermitian Diagonalization

## Before You Start

```bash
# 1. Check PETSc complex support
./check_petsc_complex.sh

# 2. If needed, recompile PETSc
cd $PETSC_DIR
./configure --with-scalar-type=complex --with-precision=double
make all test

# 3. Rebuild SLEPc
cd $SLEPC_DIR
make all test

# 4. Update environment
export PETSC_ARCH=arch-linux-c-complex
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$SLEPC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH

# 5. Rebuild Rust
cargo clean && cargo build --release
```

## Basic Usage

```rust
use light_matter::slepc::{diagonalize_slepc, verify_hermitian};
use rust_slepc_bindings::{initialize, finalize, EPSType};
use num_complex::Complex64 as c64;

fn main() {
    // Initialize
    initialize().unwrap();
    
    // Your operator (must implement SolverOperator<Complex64>)
    let op = MyOperator::new();
    
    // Optional: verify hermiticity
    assert!(verify_hermitian(&op, 1e-10));
    
    // Diagonalize
    let (evals, evecs) = diagonalize_slepc(
        &op,
        100,                          // n_eig
        128,                          // block_size
        Some(EPSType::KrylovSchur),   // solver
        None                          // initial_guess
    );
    
    // Use results
    println!("Eigenvalues: {:?}", evals);
    
    // Cleanup
    finalize().unwrap();
}
```

## Operator Implementation

Your operator must implement `SolverOperator<Complex64>`:

```rust
use num_complex::Complex64 as c64;
use light_matter::utils::SolverOperator;

struct MyOperator {
    // your data
}

impl SolverOperator<c64> for MyOperator {
    fn apply(&self, x: &[c64], y: &mut [c64]) {
        // Compute y = H * x
        // where H is your Hermitian matrix
    }
    
    fn apply_transpose(&self, x: &[c64], y: &mut [c64]) {
        // For Hermitian: H^T = H^*
        // Usually just call apply() for Hermitian matrices
        self.apply(x, y)
    }
    
    fn get_diagonal(&self) -> Vec<c64> {
        // Return diagonal elements (for preconditioning)
    }
    
    fn dimension(&self) -> usize {
        // Return matrix dimension
    }
    
    fn diagonalize(...) -> (Array1<f64>, Array2<f64>) {
        unimplemented!("Use diagonalize_slepc instead")
    }
}
```

## Testing

```bash
# Run all complex tests
cargo test --test test_complex_hermitian -- --ignored --nocapture

# Run specific test
cargo test --test test_complex_hermitian test_diagonal_hermitian -- --ignored --nocapture

# Test hermiticity
cargo test --test test_complex_hermitian test_hermiticity_check -- --ignored --nocapture
```

## Solver Types

| Solver | Best For | Notes |
|--------|----------|-------|
| `KrylovSchur` | General Hermitian | Recommended default |
| `Lanczos` | Symmetric real | Fast but less stable |
| `GeneralizedDavidson` | Interior eigenvalues | Good with preconditioner |
| `JacobiDavidson` | Difficult problems | Robust but slower |
| `ARPACK` | Legacy code | Widely tested |

## Common Issues

### Issue: Linking errors
```
undefined reference to `VecCreate_Seq_Complex`
```
**Fix**: PETSc not compiled with complex support
```bash
./check_petsc_complex.sh
```

### Issue: Wrong eigenvalues
**Fix**: Check hermiticity
```rust
verify_hermitian(&op, 1e-10);
```

### Issue: Slow convergence
**Fix**: Try different solver or increase block size
```rust
diagonalize_slepc(&op, n_eig, 256, Some(EPSType::GeneralizedDavidson), None)
```

### Issue: Memory errors
**Fix**: Check dimension consistency
```rust
assert_eq!(x.len(), self.dimension());
assert_eq!(y.len(), self.dimension());
```

## Data Format

### PETSc Complex Vector Storage
```
[re_0, im_0, re_1, im_1, ..., re_n, im_n]
```
- Size: `2*n` doubles for `n` complex numbers
- Interleaved real and imaginary parts

### Hermitian Matrix Properties
- Eigenvalues: **Real** (λ ∈ ℝ)
- Eigenvectors: Can be chosen **real** (up to phase)
- Condition: H = H† (conjugate transpose)

## Performance Tips

1. **Use Hermitian solver**: Set problem type to `Hermitian`
2. **Optimize BLAS**: Use MKL or OpenBLAS
3. **Block methods**: Increase block size for multiple eigenvalues
4. **Initial guess**: Provide good initial vectors if available
5. **Preconditioning**: Implement good `get_diagonal()`

## Verification Checklist

- [ ] PETSc compiled with `--with-scalar-type=complex`
- [ ] Environment variables set correctly
- [ ] Operator implements `SolverOperator<Complex64>`
- [ ] Hermiticity verified: `verify_hermitian(&op, 1e-10)`
- [ ] Tests pass: `cargo test --test test_complex_hermitian -- --ignored`
- [ ] Eigenvalues are real (imaginary parts ≈ 0)
- [ ] Results make physical sense

## Files to Reference

- `COMPLEX_SLEPC_MIGRATION.md` - Detailed migration guide
- `CHANGES_SUMMARY.md` - Technical details of changes
- `test_complex_hermitian.rs` - Example implementations
- `check_petsc_complex.sh` - Verification script

## Quick Debug

```rust
// Add to your code for debugging
use light_matter::slepc::verify_hermitian;

// Before diagonalization
println!("Checking hermiticity...");
if !verify_hermitian(&op, 1e-10) {
    panic!("Operator is not Hermitian!");
}

// After diagonalization
println!("Eigenvalues: {:?}", &evals[0..10.min(evals.len())]);
println!("Min eigenvalue: {}", evals.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
println!("Max eigenvalue: {}", evals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
```

## Getting Help

1. Run `./check_petsc_complex.sh` to verify installation
2. Check `COMPLEX_SLEPC_MIGRATION.md` for detailed instructions
3. Review test cases in `test_complex_hermitian.rs`
4. Add debug output to track data flow
5. Verify operator with small test cases first
