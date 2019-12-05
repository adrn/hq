#!/bin/bash
#SBATCH -J apogee-control
#SBATCH -o apogee-control.o%j
#SBATCH -e apogee-control.e%j
#SBATCH -n 720
#SBATCH -t 72:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile
init_conda
echo $HQ_RUN

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

module load gcc openmpi2 lib/hdf5/1.10.1 intel/mkl/2019-3 slurm

conda activate hq

date

stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python -m mpi4py.futures run_injected_control.py --name $HQ_RUN -v --mpi

date

