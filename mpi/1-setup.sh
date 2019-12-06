#!/bin/bash
#SBATCH -J apogee-setup
#SBATCH -o logs/apogee-setup.o%j
#SBATCH -e logs/apogee-setup.e%j
#SBATCH -n 80
#SBATCH -t 04:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile
init_conda
conda activate hq
echo $HQ_RUN

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

date

stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python3 -m mpi4py.futures make_prior_cache.py --name $HQ_RUN -v --mpiasync -o -v

stdbuf -o0 -e0 python3 make_tasks.py --name $HQ_RUN -v

date
