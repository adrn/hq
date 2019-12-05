#!/bin/bash
#SBATCH -J apogee-mcmc
#SBATCH -o apogee-mcmc.o%j
#SBATCH -e apogee-mcmc.e%j
#SBATCH -t 2:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake
#SBATCH -N 10 

source ~/.bash_profile
init_conda
conda activate hq
module load disBatch
echo $HQ_RUN

cd /mnt/ceph/users/apricewhelan/projects/hq/mpi

# python3 run_continue_mcmc.py --name $HQ_RUN --num $SLURM_ARRAY_TASK_ID -v
disBatch.py mcmc_taskfile
