#!/bin/bash
#SBATCH -J apogee-run
#SBATCH -o logs/apogee-run.o%j
#SBATCH -e logs/apogee-run.e%j
#SBATCH -n 640
#SBATCH -t 72:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile
init_conda
conda activate hq
echo $HQ_RUN

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

date

# stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python3 -m mpi4py.futures run_apogee.py --name $HQ_RUN -v --mpi
mpirun -n $SLURM_NTASKS python3 run_apogee.py --name $HQ_RUN --mpi -v

date

