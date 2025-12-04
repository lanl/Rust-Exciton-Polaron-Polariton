#!/bin/bash
#
#SBATCH --job-name=sing_ex_disp
#SBATCH --output myjob_%j.out
#SBATCH --error myjob_%j.err
#SBATCH --partition=general
#
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=64
#SBATCH --nodes=4  
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8           # 8 cores per rank
#SBATCH --mem=200G
#SBATCH --qos=long

ulimit -c 0  # Disable core dumps

# Load Intel MKL
module load intel-mkl
module load intel-mpi

# Allow 8 threads per MPI rank
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Intel MPI pinning - each rank gets 8 cores
export I_MPI_PIN_DOMAIN=omp         # Pin to OMP domain
export I_MPI_PIN=1

cargo build --release
mpirun -np 64 ../target/release/light-matter