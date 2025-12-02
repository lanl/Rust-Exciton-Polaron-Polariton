#!/bin/bash
#
#SBATCH --job-name=sing_ex_disp
#SBATCH --output myjob_%j.out
#SBATCH --error myjob_%j.err
#SBATCH --partition=general
#
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --mem=200G
#SBATCH --qos=long

ulimit -c 0  # Disable core dumps

# Load Intel MKL
module load intel-mkl
module load intel-mpi

cargo build --release
mpirun -np 24 ../target/release/light-matter