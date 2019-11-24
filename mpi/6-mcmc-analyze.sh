#!/bin/bash
#SBATCH -J apogee-analyze-mcmc
#SBATCH -o apogee-analyze-mcmc.o%j
#SBATCH -e apogee-analyze-mcmc.e%j
#SBATCH -n 40
#SBATCH -t 16:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake
# --dependency=afterok:257249

source ~/.bash_profile
init_env
echo $HQ_RUN

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

date

stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python3 -m mpi4py.futures analyze_emcee_samplings.py --name $HQ_RUN -v --mpi

date
