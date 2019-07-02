#!/bin/bash
#SBATCH -J apogee-analyze-emcee
#SBATCH -o apogee-analyze-emcee.o%j
#SBATCH -e apogee-analyze-emcee.e%j
#SBATCH -n 40
#SBATCH -t 16:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake
# --dependency=afterok:257249

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

module load gcc openmpi2 lib/hdf5/1.10.1 intel/mkl/2019-3 slurm

conda activate hq

date

stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python -m mpi4py.futures analyze_emcee_samplings.py --name $HQ_RUN -v --mpi

date
