#!/bin/bash
#SBATCH -J load
#SBATCH -o load.o%j
#SBATCH -e load.e%j
#SBATCH -N 8
#SBATCH -t 24:00:00
#SBATCH -p cca

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

module load gcc openmpi2

conda activate hq

date

srun python run_apogee.py -v -c ../run_config/apogee.yml --mpi

date
