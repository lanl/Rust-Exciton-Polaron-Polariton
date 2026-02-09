# Segmentation Fault Fix for MPI Jobs

## Problem

When running the light-matter simulation with multiple MPI ranks using `sbatch submit.sh`, the program would complete the computation successfully but then crash with segmentation faults during finalization:

```
[0]PETSC ERROR: Caught signal number 11 SEGV: Segmentation Violation, probably memory access out of range
MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD
```

## Root Cause

The issue was caused by **unsynchronized finalization of PETSc/SLEPc across MPI ranks**:

1. Each MPI rank runs independently and completes its work at different times
2. The program uses file-based barriers for synchronization during computation
3. **However, there was NO barrier before calling `finalize()`**
4. When ranks finish at different times and try to finalize PETSc/SLEPc independently, it causes race conditions and segmentation faults

## Solution

Added a **file-based MPI barrier synchronization** before calling `finalize()` in `main()`:

### Key Changes in `src/main.rs`:

1. **Added `mpi_barrier_sync()` function** (lines 45-100):
   - Reads MPI rank and world size from environment variables
   - Creates a unique barrier directory using the SLURM job ID
   - Each rank writes a barrier file when it reaches the synchronization point
   - All ranks wait until all barrier files exist
   - Rank 0 cleans up the barrier directory after synchronization

2. **Added barrier call before finalization** (lines 730-733):
   ```rust
   // Synchronize all MPI ranks before finalizing SLEPc/PETSc
   // This is CRITICAL to prevent segmentation faults during finalization
   timed_println!("Synchronizing MPI ranks before finalization...");
   mpi_barrier_sync();
   timed_println!("All ranks synchronized, finalizing SLEPc...");
   
   // Finalize SLEPc
   finalize().expect("Failed to finalize SLEPc");
   ```

## Why This Works

1. **Ensures all ranks complete their work** before any rank attempts to finalize
2. **Prevents race conditions** in PETSc/SLEPc cleanup routines
3. **Uses file-based synchronization** which is compatible with the existing MPI setup (no need for MPI library calls)
4. **Handles single-rank case** gracefully (no synchronization needed when world_size == 1)

## Testing

To verify the fix works:

```bash
cd light-matter
sbatch submit.sh
```

The job should now complete without segmentation faults. You should see in the output:

```
Synchronizing MPI ranks before finalization...
All ranks synchronized, finalizing SLEPc...
Job completed.
```

## Additional Notes

- The barrier uses `/tmp/mpi_barrier_{job_id}` for temporary synchronization files
- The barrier directory is automatically cleaned up by rank 0 after synchronization
- This approach is compatible with Intel MPI, OpenMPI, and other MPI implementations
- The fix does not affect single-process runs (world_size == 1)

## Related Files

- `src/main.rs` - Main program with barrier synchronization
- `src/color_band_plot.rs` - Already uses file-based barriers for computation
- `submit.sh` - SLURM submission script (no changes needed)
