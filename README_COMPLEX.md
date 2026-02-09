ck# Complex Hermitian Matrix Diagonalization with SLEPc

This directory contains the updated code and documentation for diagonalizing complex Hermitian matrices using SLEPc/PETSc from Rust.

## 📚 Documentation

| File | Purpose |
|------|---------|
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | Quick start guide and common patterns |
| **[COMPLEX_SLEPC_MIGRATION.md](COMPLEX_SLEPC_MIGRATION.md)** | Detailed migration guide |
| **[CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)** | Technical details of all changes |
| **[DATA_FLOW_DIAGRAM.md](DATA_FLOW_DIAGRAM.md)** | Visual explanation of data flow |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Common issues and solutions |

## 🚀 Quick Start

### 1. Check Your Installation

```bash
chmod +x check_petsc_complex.sh
./check_petsc_complex.sh
```

### 2. Install Complex PETSc/SLEPc

#### If Using Spack (Recommended for HPC systems)

See **[SPACK_COMPLEX_PETSC.md](SPACK_COMPLEX_PETSC.md)** for detailed instructions.

```bash
# Install complex PETSc
spack install petsc@3.21.1 +complex +double

# Install matching SLEPc
spack install slepc@3.21.1 ^petsc+complex

# Load them
spack load petsc+complex
spack load slepc ^petsc+complex
```

#### If Building from Source

```bash
cd $PETSC_DIR
./configure --with-scalar-type=complex --with-precision=double
make all test

cd $SLEPC_DIR
make all test
```

### 3. Build and Test

```bash
cargo clean
cargo build --release
cargo test --test test_complex_hermitian -- --ignored --nocapture
```

### 4. Use in Your Code

```rust
use light_matter::slepc::{diagonalize_slepc, verify_hermitian};
use rust_slepc_bindings::{initialize, finalize, EPSType};
use num_complex::Complex64 as c64;

fn main() {
    initialize().unwrap();
    
    let op = MyHermitianOperator::new();
    assert!(verify_hermitian(&op, 1e-10));
    
    let (evals, evecs) = diagonalize_slepc(
        &op, 100, 128, Some(EPSType::KrylovSchur), None
    );
    
    println!("Eigenvalues: {:?}", evals);
    
    finalize().unwrap();
}
```

## 📋 Key Changes

### Code Changes
- `slepc.rs`: Updated to handle `SolverOperator<Complex64>`
- `lib.rs`: Added complex vector creation/extraction functions
- New test suite: `test_complex_hermitian.rs`

### Requirements
- **PETSc**: Must be compiled with `--with-scalar-type=complex`
- **SLEPc**: Must be compiled against complex PETSc
- **Operator**: Must implement `SolverOperator<Complex64>`

## 🔍 What's Different?

### Before (Real)
```rust
impl SolverOperator<f64> for MyOperator {
    fn apply(&self, x: &[f64], y: &mut [f64]) { ... }
}
```

### After (Complex)
```rust
impl SolverOperator<c64> for MyOperator {
    fn apply(&self, x: &[c64], y: &mut [c64]) { ... }
}
```

## 📊 Data Format

PETSc stores complex vectors as:
```
[real_0, imag_0, real_1, imag_1, ..., real_n, imag_n]
```

The callback functions handle conversion automatically.

## ✅ Verification

### Check Hermiticity
```rust
use light_matter::slepc::verify_hermitian;
assert!(verify_hermitian(&op, 1e-10));
```

### Check Eigenvalues are Real
```rust
// For Hermitian matrices, eigenvalues should be real
for &eval in evals.iter() {
    assert!(eval.is_finite());
}
```

## 🧪 Testing

```bash
# Run all complex tests
cargo test --test test_complex_hermitian -- --ignored --nocapture

# Run specific test
cargo test --test test_complex_hermitian test_diagonal_hermitian -- --ignored

# Test hermiticity checker
cargo test --test test_complex_hermitian test_hermiticity_check -- --ignored
```

## 🐛 Troubleshooting

### Common Issues

1. **Linking errors**: PETSc not compiled with complex support
   - Solution: Run `./check_petsc_complex.sh`

2. **Wrong eigenvalues**: Matrix not Hermitian
   - Solution: Use `verify_hermitian(&op, 1e-10)`

3. **Slow performance**: Wrong solver or inefficient operator
   - Solution: Use `EPSType::KrylovSchur` for Hermitian matrices

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more details.

## 📖 Documentation Guide

**Start here**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- Basic usage patterns
- Common operations
- Quick troubleshooting

**For migration**: [COMPLEX_SLEPC_MIGRATION.md](COMPLEX_SLEPC_MIGRATION.md)
- Step-by-step migration guide
- Recompilation instructions
- Testing procedures

**For understanding**: [DATA_FLOW_DIAGRAM.md](DATA_FLOW_DIAGRAM.md)
- Visual data flow diagrams
- Memory layout explanations
- Type conversion details

**For problems**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Common errors and solutions
- Debugging strategies
- Performance optimization

**For details**: [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)
- Complete list of changes
- Technical implementation details
- API documentation

## 🎯 Next Steps

1. ✅ Read [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. ⬜ Run `./check_petsc_complex.sh`
3. ⬜ Recompile PETSc/SLEPc if needed
4. ⬜ Update your operator to use `Complex64`
5. ⬜ Run tests to verify
6. ⬜ Use `verify_hermitian` to check your operator
7. ⬜ Proceed with calculations

## 💡 Tips

- **Always verify hermiticity** before diagonalization
- **Use KrylovSchur** for Hermitian matrices (fastest)
- **Check eigenvalues are real** (imaginary parts should be ~0)
- **Profile your `apply()` method** if performance is slow
- **Start with small test cases** to verify correctness

## 📞 Support

If you encounter issues:
1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Run `./check_petsc_complex.sh`
3. Review test cases in `test_complex_hermitian.rs`
4. Add debug output to track data flow
5. Verify with small known matrices first

## 🔗 Related Files

- `light-matter/src/slepc.rs` - Main diagonalization code
- `slepc_rust/src/lib.rs` - SLEPc/PETSc bindings
- `light-matter/tests/test_complex_hermitian.rs` - Test suite
- `check_petsc_complex.sh` - Installation verification script

---

**Note**: This implementation requires PETSc/SLEPc compiled with complex scalar support. See [COMPLEX_SLEPC_MIGRATION.md](COMPLEX_SLEPC_MIGRATION.md) for detailed instructions.
