#!/bin/bash
#SBATCH -J apogee-mcmc
#SBATCH -o logs/apogee-mcmc.o%j
#SBATCH -e logs/apogee-mcmc.e%j
#SBATCH -t 36:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake
#SBATCH -N 10 

module load disBatch
disBatch.py mcmc_taskfile
