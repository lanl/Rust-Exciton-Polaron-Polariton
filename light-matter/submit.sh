#!/bin/bash
#
#SBATCH --job-name=sing_ex_disp
#SBATCH --output myjob_%j.out
#SBATCH --error myjob_%j.err
#SBATCH --partition=general
#
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=100G
#SBATCH --qos=long

ulimit -c 0  # Disable core dumps

# Load Intel MKL
module load intel-mkl

cargo run --release