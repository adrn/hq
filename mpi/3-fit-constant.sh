#!/bin/bash
#SBATCH -J apogee-const
#SBATCH -o logs/apogee-const.o%j
#SBATCH -e logs/apogee-const.e%j
#SBATCH -n 400
#SBATCH -t 24:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile
init_conda
conda activate hq
echo $HQ_RUN

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

date

mpirun -n $SLURM_NTASKS python3 run_fit_constant.py --name $HQ_RUN -v --mpi -o

date

# --constraint=skylake
