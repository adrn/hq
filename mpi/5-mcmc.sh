#!/bin/bash
#SBATCH -J apogee-mcmc
#SBATCH -o apogee-mcmc.o%j
#SBATCH -e apogee-mcmc.e%j
#SBATCH -n 400
#SBATCH -t 12:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile
init_env
echo $HQ_RUN

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

date

stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python3 -m mpi4py.futures run_continue_mcmc.py --name $HQ_RUN -v --mpi

date
