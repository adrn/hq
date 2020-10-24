#!/bin/bash
#SBATCH -J apogee-inject
#SBATCH -o logs/apogee-inject.o%j
#SBATCH -e logs/apogee-inject.e%j
#SBATCH -n 80
#SBATCH -t 10:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile
init_conda
conda activate hq
echo $HQ_RUN

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

date

mpirun -n $SLURM_NTASKS python3 make_injected_data.py --name $HQ_RUN --mpi -v

date

