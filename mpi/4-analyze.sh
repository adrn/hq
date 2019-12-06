#!/bin/bash
#SBATCH -J apogee-analyze
#SBATCH -o logs/apogee-analyze.o%j
#SBATCH -e logs/apogee-analyze.e%j
#SBATCH -n 80
#SBATCH -t 16:00:00
#SBATCH -p gen
#SBATCH --constraint=skylake

source ~/.bash_profile
init_conda
conda activate hq
echo $HQ_RUN

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

date

stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python3  analyze_joker_samplings.py --name $HQ_RUN -v --mpi

date
