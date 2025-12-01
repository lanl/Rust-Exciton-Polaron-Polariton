MPI PARALLELIZATION - QUICK START
==================================

WHAT IT IS
----------
Band structure calculations parallelized using MPI (Message Passing Interface).
Uses proper Rust `mpi` crate for direct memory communication.

WHY NOT THREADING (RAYON)?
---------------------------
PETSc/SLEPc is NOT thread-safe. Threading causes crashes.
MPI uses separate processes, which works perfectly.

INSTALL (One-time)
------------------
Ubuntu/Debian:  sudo apt-get install libopenmpi-dev openmpi-bin
CentOS/RHEL:    sudo yum install openmpi-devel
macOS:          brew install open-mpi

BUILD
-----
cargo build --release

RUN
---
Sequential:     cargo run --release
MPI (4 ranks):  mpirun -np 4 ./target/release/light-matter
MPI (8 ranks):  mpirun -np 8 ./target/release/light-matter

WHEN TO USE WHAT
----------------
< 100 k-points   → Use sequential
> 200 k-points   → Use MPI

DOCUMENTATION
-------------
See MPI_GUIDE.md for complete documentation.

CLUSTER
-------
Example SLURM:  sbatch examples/submit_slurm.sh
Example PBS:    qsub examples/submit_pbs.sh

TROUBLESHOOTING
---------------
1. Test sequential first: cargo run --release
2. Test with 1 rank:      mpirun -np 1 ./target/release/light-matter
3. Scale up:              mpirun -np 4 ./target/release/light-matter
4. See MPI_GUIDE.md for more help
