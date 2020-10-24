#!/bin/bash
#SBATCH -J apogee-control
#SBATCH -o logs/null-control.o%j
#SBATCH -e logs/null-control.e%j
#SBATCH -n 400
#SBATCH -t 36:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile
init_conda
conda activate hq3
echo $HQ_RUN

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts


date

# stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python -m mpi4py.futures run_null_control.py --name $HQ_RUN -v --mpi
mpirun -n $SLURM_NTASKS python3 run_null_control.py --name $HQ_RUN --mpi -v

date

