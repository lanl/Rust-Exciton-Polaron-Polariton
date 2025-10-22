#!/bin/bash
#
#SBATCH --job-name=sing_ex_disp
#SBATCH --output myjob_%j.out
#SBATCH --error myjob_%j.err
#SBATCH --partition=general
#
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --mem=128
#SBATCH --qos=long

# Load Intel MKL
module load intel-mkl

./build.sh run --release