#!/bin/bash
#SBATCH -J apogee-const
#SBATCH -o apogee-const.o%j
#SBATCH -e apogee-const.e%j
#SBATCH -n 720
#SBATCH -t 72:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

conda activate hq

date

stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python -m mpi4py.futures run_fit_constant.py --name $HQ_RUN -v --mpi -o

date

# --constraint=skylake
