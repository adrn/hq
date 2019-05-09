#!/bin/bash
#SBATCH -J apogee
#SBATCH -o apogee.o%j
#SBATCH -e apogee.e%j
#SBATCH -N 8
#SBATCH -t 72:00:00
#SBATCH -p cca

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

module load gcc openmpi2 lib/hdf5/1.10.1 intel/mkl/2019-3

conda activate hq

date

mpirun python run_apogee.py -v -c ../run_config/apogee.yml --mpi --overwrite

date
