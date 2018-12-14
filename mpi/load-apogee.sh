#!/bin/bash
#SBATCH -J apogee-load          # job name
#SBATCH -o apogee-load.o%j             # output file name (%j expands to jobID)
#SBATCH -e apogee-load.e%j             # error file name (%j expands to jobID)
#SBATCH --mem=16000
#SBATCH --ntasks=1
#SBATCH -t 06:00:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/twoface/scripts/

source activate twoface

date

python initdb.py --allstar=../data/APOGEE_DR14/allStar-l31c.2.fits --allvisit=../data/APOGEE_DR14/allVisit-l31c.2.fits -v

date
