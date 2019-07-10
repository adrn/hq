#!/bin/bash
#SBATCH -J apogee-emcee
#SBATCH -o apogee-emcee.o%j
#SBATCH -e apogee-emcee.e%j
#SBATCH -n 400
#SBATCH -t 12:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

conda activate hq

date

stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python -m mpi4py.futures run_continue_mcmc.py --name $HQ_RUN -v --mpi

date
