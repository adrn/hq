#!/bin/bash
#SBATCH -J apogee-run
#SBATCH -o apogee-run.o%j
#SBATCH -e apogee-run.e%j
#SBATCH -n 720
#SBATCH -t 72:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

conda activate hq

date

stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python -m mpi4py.futures run_apogee.py --name $HQ_RUN -v --mpi

date

# --constraint=skylake
