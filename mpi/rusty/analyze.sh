#!/bin/bash
#SBATCH -J apogee-analyze
#SBATCH -o apogee-analyze.o%j
#SBATCH -e apogee-analyze.e%j
#SBATCH -n 1
#SBATCH -t 16:00:00
#SBATCH -p cca
#SBATCH --dependency=afterok:257249

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

module load gcc openmpi2 lib/hdf5/1.10.1 intel/mkl/2019-3

conda activate hq

date

python analyze_samplings.py --name dr16-beta-snr-jitter -v

date
