#!/bin/bash
#SBATCH -J apogee-setup
#SBATCH -o apogee-setup.o%j
#SBATCH -e apogee-setup.e%j
#SBATCH -n 400
#SBATCH -t 18:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile
init_env
echo $HQ_RUN

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

date

stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python3 -m mpi4py.futures make_prior_cache.py --name $HQ_RUN -v --mpi

stdbuf -o0 -e0 python3 make_tasks.py --name $HQ_RUN -v

date
