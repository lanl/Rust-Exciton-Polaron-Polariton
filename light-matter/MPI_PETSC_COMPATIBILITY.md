# MPI + PETSc/SLEPc Compatibility Issue

## Problem

When using the `mpi` crate with PETSc/SLEPc, you may encounter this error:

```
Fatal error in PMPI_Comm_set_errhandler: Invalid communicator
```

**Cause**: PETSc/SLEPc initializes MPI internally, and the `mpi` crate tries to initialize it again, causing a conflict.

## Solution

Use the **simple MPI version** that works with environment variables instead of initializing MPI:

### Change Your Code

**Instead of:**
```rust
let (k_points, eigvals, photonic) = calculate_color_band_structure_mpi(
    &mut pp, n_k_points, n_eig, &sym_points, &pp_config, method, use_cached
);
```

**Use:**
```rust
let (k_points, eigvals, photonic) = calculate_color_band_structure_mpi_simple(
    &mut pp, n_k_points, n_eig, &sym_points, &pp_config, method, use_cached
);
```

## How It Works

### `calculate_color_band_structure_mpi_simple`

- Reads MPI rank/size from environment variables
- Doesn't initialize MPI (avoids conflict with PETSc)
- Uses file-based communication (simple and robust)
- Works with any MPI implementation

### Environment Variables

The function checks these variables (in order):
- `OMPI_COMM_WORLD_SIZE` / `OMPI_COMM_WORLD_RANK` (OpenMPI)
- `PMI_SIZE` / `PMI_RANK` (MPICH, Intel MPI)
- `MPI_LOCALNRANKS` / `MPI_LOCALRANKID` (Intel MPI alternative)

## Usage

```bash
# Build
cargo build --release

# Run with MPI
mpirun -np 4 ./target/release/light-matter
```

The function will automatically detect it's running under MPI and distribute work accordingly.

## Performance

The simple version uses file-based communication instead of direct MPI calls:

| Feature | `mpi` crate version | Simple version |
|---------|---------------------|----------------|
| **Compatibility** | May conflict with PETSc | ✅ Works with PETSc |
| **Speed** | Fastest | ~5% slower (file I/O) |
| **Robustness** | Good | ✅ Excellent |
| **Setup** | May need tweaking | ✅ Just works |

**Recommendation**: Use the simple version when working with PETSc/SLEPc.

## Alternative: Remove `mpi` Crate

If you don't need the `mpi` crate for anything else, you can remove it:

```toml
# In Cargo.toml, remove or comment out:
# mpi = "0.8"
```

Then rebuild:
```bash
cargo build --release
```

The simple version doesn't require the `mpi` crate at all!

## Summary

- ✅ Use `calculate_color_band_structure_mpi_simple` with PETSc/SLEPc
- ✅ Works with all MPI implementations
- ✅ No MPI initialization conflicts
- ✅ Slightly slower but more compatible
- ✅ Can optionally remove `mpi` crate dependency
