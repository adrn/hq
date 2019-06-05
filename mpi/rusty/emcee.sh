#!/bin/bash
#SBATCH -J apogee-emcee
#SBATCH -o apogee-emcee.o%j
#SBATCH -e apogee-emcee.e%j
#SBATCH -n 400
#SBATCH -t 12:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

module load gcc openmpi2 lib/hdf5/1.10.1 intel/mkl/2019-3

conda activate hq

date

mpirun -n $SLURM_NTASKS python -m mpi4py.futures run_continue_mcmc.py --name dr16-beta-snr-jitter -v --mpi

date
