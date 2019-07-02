#!/bin/bash
#SBATCH -J apogee-tasks
#SBATCH -o apogee-tasks.o%j
#SBATCH -e apogee-tasks.e%j
#SBATCH -n 1
#SBATCH -t 16:00:00
#SBATCH -p cca

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

module load gcc openmpi2 lib/hdf5/1.10.1 intel/mkl/2019-3 slurm

conda activate hq

date

stdbuf -o0 -e0 python make_tasks.py --name $HQ_RUN -v

date
