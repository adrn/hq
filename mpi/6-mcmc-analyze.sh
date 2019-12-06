#!/bin/bash
#SBATCH -J apogee-analyze-mcmc
#SBATCH -o logs/apogee-analyze-mcmc.o%j
#SBATCH -e logs/apogee-analyze-mcmc.e%j
#SBATCH -n 40
#SBATCH -t 16:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile
init_conda
conda activate hq2
echo $HQ_RUN

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

date

stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python3 analyze_mcmc_samplings.py --name $HQ_RUN -v --mpi

date
