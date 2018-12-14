#!/bin/bash
#SBATCH -J exp2          # job name
#SBATCH -o exp2.o%j             # output file name (%j expands to jobID)
#SBATCH -e exp2.e%j             # error file name (%j expands to jobID)
#SBATCH -n 224
#SBATCH -t 16:00:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/twoface/scripts/exp2-num-epochs/

module load openmpi/gcc/1.10.2/64

source activate twoface

date

srun python exp2.py -v --mpi --circ --time_sampling='uniform' -P 128

srun python exp2.py -v --mpi --time_sampling='uniform' -P 8
srun python exp2.py -v --mpi --circ --time_sampling='uniform' -P 8
srun python exp2.py -v --mpi --time_sampling='log' -P 8
srun python exp2.py -v --mpi --circ --time_sampling='log' -P 8

srun python exp2.py -v --mpi --time_sampling='uniform' -P 2
srun python exp2.py -v --mpi --circ --time_sampling='uniform' -P 2
srun python exp2.py -v --mpi --time_sampling='log' -P 2
srun python exp2.py -v --mpi --circ --time_sampling='log' -P 2


date
