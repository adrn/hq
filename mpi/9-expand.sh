#!/bin/bash
#SBATCH -J apogee-expand
#SBATCH -o logs/expand.o%j
#SBATCH -e logs/expand.e%j
#SBATCH -N 4
#SBATCH -t 12:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile
init_conda

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

conda activate hq

date

mpirun -n $SLURM_NTASKS python3 expand_samples.py --name $HQ_RUN -v --mpi

cd /mnt/ceph/users/apricewhelan/projects/hq/cache/$HQ_RUN
tar -czf samples.tar.gz samples/

date
