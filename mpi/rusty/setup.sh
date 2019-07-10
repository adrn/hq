#!/bin/bash
#SBATCH -J apogee-setup
#SBATCH -o apogee-setup.o%j
#SBATCH -e apogee-setup.e%j
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -p cca

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/hq/scripts

conda activate hq

date

stdbuf -o0 -e0 python make_prior_cache.py --name $HQ_RUN -v

stdbuf -o0 -e0 python make_tasks.py --name $HQ_RUN -v

date
