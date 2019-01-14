#!/bin/bash
#SBATCH -J apogee-load
#SBATCH -o apogee-load.o%j
#SBATCH -e apogee-load.e%j
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -p cca

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

module load gcc openmpi2 lib/hdf5/1.10.1

conda activate hq

date

python db_init.py --allstar=../data/allStar-t9-l31c-58247.fits --allvisit=../data/allVisit-t9-l31c-58247.fits -v

date
