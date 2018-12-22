#!/bin/bash
#SBATCH -J load
#SBATCH -o load.o%j
#SBATCH -e load.e%j
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p cca

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

conda activate hq

date

python init_db.py --allstar=../data/allStar-t9-l31c-58247.fits --allvisit=../data/allVisit-t9-l31c-58247.fits -v

date
