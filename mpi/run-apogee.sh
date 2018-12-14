#!/bin/bash
#SBATCH -J apogee          # job name
#SBATCH -o apogee.o%j             # output file name (%j expands to jobID)
#SBATCH -e apogee.e%j             # error file name (%j expands to jobID)
#SBATCH -n 448
#SBATCH -t 48:00:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/twoface/scripts/

module load openmpi/gcc/1.10.2/64

source activate twoface

date

srun python run_apogee.py -v -c ../run_config/apogee.yml --mpi

date
