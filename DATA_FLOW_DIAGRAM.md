# Data Flow Diagram: Complex Hermitian Diagonalization

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Your Rust Application                        │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  MyOperator: SolverOperator<Complex64>                 │    │
│  │  - apply(&self, x: &[c64], y: &mut [c64])             │    │
│  │  - dimension(&self) -> usize                           │    │
│  └────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            │ diagonalize_slepc()                │
│                            ▼                                     │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │
┌─────────────────────────────────────────────────────────────────┐
│                    slepc.rs (Rust Wrapper)                       │
│                                                                   │
│  1. Create shell matrix                                          │
│  2. Set mv_product_complex as callback                          │
│  3. Configure solver (Hermitian, tolerances, etc.)              │
│  4. Call EPSSolve()                                             │
│  5. Extract eigenvalues & eigenvectors                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ FFI calls
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SLEPc/PETSc (C Library)                        │
│                                                                   │
│  - Iterative eigenvalue solver                                   │
│  - Calls mv_product_complex for each iteration                  │
│  - Manages convergence, orthogonalization, etc.                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Matrix-Vector Product Callback Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         SLEPc/PETSc                              │
│                                                                   │
│  Needs to compute: y = H * x                                     │
│  Calls: mv_product_complex(mat, x_petsc, y_petsc)              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ x_petsc, y_petsc (PETSc Vec)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              mv_product_complex (Callback Function)              │
│                                                                   │
│  Step 1: Extract operator from context                          │
│  ┌────────────────────────────────────────────────────┐        │
│  │ MatShellGetContext(mat, &mut ctx_ptr)              │        │
│  │ op = &*(ctx_ptr as *const T)                       │        │
│  └────────────────────────────────────────────────────┘        │
│                                                                   │
│  Step 2: Get raw arrays from PETSc vectors                      │
│  ┌────────────────────────────────────────────────────┐        │
│  │ VecGetArray(x_petsc, &mut x_array)                 │        │
│  │ VecGetArray(y_petsc, &mut y_array)                 │        │
│  │                                                      │        │
│  │ x_array: [re0, im0, re1, im1, ..., re_n, im_n]    │        │
│  │          └─────┬─────┘                              │        │
│  │         Interleaved format (2*n doubles)           │        │
│  └────────────────────────────────────────────────────┘        │
│                                                                   │
│  Step 3: Convert to Rust Complex64                              │
│  ┌────────────────────────────────────────────────────┐        │
│  │ for i in 0..n {                                     │        │
│  │   x_complex[i] = c64::new(                         │        │
│  │     x_array[2*i],      // real part                │        │
│  │     x_array[2*i + 1]   // imag part                │        │
│  │   )                                                 │        │
│  │ }                                                   │        │
│  │                                                      │        │
│  │ x_complex: [c64, c64, ..., c64]                    │        │
│  │             └─┬─┘                                   │        │
│  │         n Complex64 values                          │        │
│  └────────────────────────────────────────────────────┘        │
│                                                                   │
│  Step 4: Call operator                                          │
│  ┌────────────────────────────────────────────────────┐        │
│  │ op.apply(&x_complex, &mut y_complex)               │        │
│  └────────────────────────────────────────────────────┘        │
│                            │                                     │
│                            ▼                                     │
│  ┌────────────────────────────────────────────────────┐        │
│  │         Your Operator's apply() Method              │        │
│  │                                                      │        │
│  │  Computes: y = H * x                                │        │
│  │  where H is your Hermitian matrix                   │        │
│  └────────────────────────────────────────────────────┘        │
│                            │                                     │
│                            ▼                                     │
│  Step 5: Convert back to interleaved format                     │
│  ┌────────────────────────────────────────────────────┐        │
│  │ for i in 0..n {                                     │        │
│  │   y_array[2*i]     = y_complex[i].re               │        │
│  │   y_array[2*i + 1] = y_complex[i].im               │        │
│  │ }                                                   │        │
│  └────────────────────────────────────────────────────┘        │
│                                                                   │
│  Step 6: Restore arrays                                         │
│  ┌────────────────────────────────────────────────────┐        │
│  │ VecRestoreArray(x_petsc, &mut x_array)             │        │
│  │ VecRestoreArray(y_petsc, &mut y_array)             │        │
│  └────────────────────────────────────────────────────┘        │
│                                                                   │
│  Return 0 (success)                                             │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ y_petsc updated
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SLEPc/PETSc                              │
│                                                                   │
│  Continues iteration with updated y                              │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Layout

### PETSc Complex Vector (Interleaved)
```
Index:  0    1    2    3    4    5   ...  2n-2  2n-1
       ┌────┬────┬────┬────┬────┬────┬───┬─────┬─────┐
Data:  │re0 │im0 │re1 │im1 │re2 │im2 │...│re_n │im_n │
       └────┴────┴────┴────┴────┴────┴───┴─────┴─────┘
        └──┬──┘  └──┬──┘  └──┬──┘       └────┬────┘
         c64[0]   c64[1]   c64[2]          c64[n]
```

### Rust Complex64 Vector
```
Index:  0         1         2       ...    n
       ┌─────────┬─────────┬─────────┬───┬─────────┐
Data:  │ c64(re0,│ c64(re1,│ c64(re2,│...│c64(re_n,│
       │    im0) │    im1) │    im2) │   │   im_n) │
       └─────────┴─────────┴─────────┴───┴─────────┘
```

## Hermitian Matrix Structure

For a Hermitian matrix H:
```
H = H†  (conjugate transpose)

     ┌                           ┐
     │  h00   h01   h02   ...    │
     │  h01*  h11   h12   ...    │
H =  │  h02*  h12*  h22   ...    │
     │  ...   ...   ...   ...    │
     └                           ┘

where:
- Diagonal elements are real: h_ii ∈ ℝ
- Off-diagonal: h_ij = h_ji* (conjugate)
```

## Eigenvalue Problem

```
H * v = λ * v

where:
- H: Hermitian matrix (n × n)
- v: Eigenvector (complex, length n)
- λ: Eigenvalue (REAL for Hermitian matrices)
```

## Complete Workflow

```
1. Initialize
   ├─ initialize() 
   └─ Create operator

2. Verify (optional)
   └─ verify_hermitian(&op, 1e-10)

3. Diagonalize
   ├─ diagonalize_slepc(&op, n_eig, ...)
   │  ├─ Create shell matrix
   │  ├─ Set mv_product_complex callback
   │  ├─ Configure solver
   │  │  ├─ Problem type: Hermitian
   │  │  ├─ Which: Smallest real
   │  │  ├─ Tolerances
   │  │  └─ Dimensions
   │  ├─ EPSSolve()
   │  │  └─ Iteratively calls mv_product_complex
   │  │     ├─ Convert PETSc → Rust
   │  │     ├─ op.apply()
   │  │     └─ Convert Rust → PETSc
   │  └─ Extract results
   │     ├─ Eigenvalues (real)
   │     └─ Eigenvectors (real parts)
   └─ Return (eigenvalues, eigenvectors)

4. Use results
   └─ Process eigenvalues and eigenvectors

5. Cleanup
   └─ finalize()
```

## Type Conversions

```
PETSc Vec (C)          Rust Slice           Rust Complex Vec
─────────────          ──────────           ────────────────
*mut c_double    →    &[f64]          →    Vec<Complex64>
(2*n elements)        (2*n elements)       (n elements)

Example for n=3:
[1.0, 2.0,      →    [1.0, 2.0,      →    [c64(1.0, 2.0),
 3.0, 4.0,            3.0, 4.0,            c64(3.0, 4.0),
 5.0, 6.0]            5.0, 6.0]            c64(5.0, 6.0)]
```

## Error Handling

```
┌─────────────────────────────────────────┐
│ PETSc/SLEPc returns error code          │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ Check: result != PETSC_SUCCESS?         │
└─────────────────┬───────────────────────┘
                  │
         ┌────────┴────────┐
         │                 │
         ▼                 ▼
    ┌────────┐      ┌──────────┐
    │ Error  │      │ Success  │
    │ Return │      │ Continue │
    │ Err()  │      │          │
    └────────┘      └──────────┘
```

## Performance Considerations

```
Operation                    Cost
─────────────────────────────────────────
Complex multiplication       ~4× real
Complex addition            ~2× real
Memory usage                ~2× real
Vector conversion           O(n)
Matrix-vector product       O(n²) or O(nnz)
Full diagonalization        O(n³)
Iterative (k eigenvalues)   O(k × n × nnz)
```

## Key Points

1. **Interleaved Format**: PETSc stores complex as [re, im, re, im, ...]
2. **Hermitian Property**: Eigenvalues are guaranteed real
3. **Callback Function**: Bridges PETSc C code and Rust operator
4. **Type Safety**: Rust's type system ensures correct usage
5. **Zero-Copy**: Direct pointer access for efficiency
6. **Error Handling**: Proper cleanup on errors
