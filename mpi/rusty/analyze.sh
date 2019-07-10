#!/bin/bash
#SBATCH -J apogee-analyze
#SBATCH -o apogee-analyze.o%j
#SBATCH -e apogee-analyze.e%j
#SBATCH -n 40
#SBATCH -t 16:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake
# --dependency=afterok:257249

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

conda activate hq

date

stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python -m mpi4py.futures analyze_joker_samplings.py --name $HQ_RUN -v --mpi

date
