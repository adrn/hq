#!/bin/bash
#SBATCH -J apogee-run
#SBATCH -o apogee-run.o%j
#SBATCH -e apogee-run.e%j
#SBATCH -n 720
#SBATCH -t 06:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile
init_env
echo $HQ_RUN

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

date

export NTASKS=$((SLURM_NTASKS))

# stdbuf -o0 -e0 mpirun -n $SLURM_NTASKS python3 -m mpi4py.futures run_apogee.py --name $HQ_RUN -v --mpi
stdbuf -o0 -e0 mpirun -n $NTASKS python3 run_apogee.py --name $HQ_RUN -v --mpi

date

# --constraint=skylake
