#!/bin/bash
#SBATCH -J zxyTest
#SBATCH -p Hygon_Z100
##SBATCH -N 1
##SBATCH -w node24
#SBATCH -c 32 
#SBATCH --gres=dcu:1

# module purge
# module add compiler/gcc/7.3.1 mpi/openmpi/gnu/4.0.3
# module add compiler/dtk/23.10.1
# source $HOME/miniconda3/etc/profile.d/conda.sh
# conda activate torch1.13-dtk2310-py310

# Perform the following directly using cuda
python testRealB.py