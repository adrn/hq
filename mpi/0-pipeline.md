HQ Pipeline on Rusty
====================

* Check your `$HQ_RUN` and `$HQ_CACHE_PATH` environment variables
  variables. `$HQ_RUN` should be set to the name of the current run of
  HQ, e.g.:

      export HQ_RUN=dr16

  and `$HQ_CACHE_PATH` should be set to the path you want HQ to cache
  results and temporary files to, e.g.:

      export HQ_CACHE_PATH=/mnt/ceph/users/apricewhelan/projects/hq/cache/

* Make sure you are using the right Python environment. This is now
  the conda environment `hq`:

      conda activate hq

* Change to `hq/scripts` and create the run of The Joker to initialize
  the run cache directory:

      python make_run.py --name $HQ_RUN

* Edit the run config file to change parameters, found in:

     $HQ_CACHE_PATH/$HQ_RUN/config.yml

* Edit the run prior specification, found in:

     $HQ_CACHE_PATH/$HQ_RUN/prior.py

* Change to `hq/mpi` and run the setup script to create a prior cache
  file and make a tasks file with a better encapsulation of the data:

     sbatch 1-setup.sh

* Run The Joker on all of the APOGEE tasks:

     sbatch 2-run.sh

* sbatch 3-fit-constant.sh

* sbatch 4-analyze.sh

* How many unimodal stars? Run an interactive job with:

      python run_continue_mcmc.py --name $HQ_RUN -v

  and edit the `mcmc_taskfile` with this number of repeat tasks.

* Run the MCMC shite:
      
      module load disBatch
      sbatch -N10 --constraint=skylake -p cca disBatch.py mcmc_taskfile

* sbatch 6-mcmc-analyze.sh

* TODO: combine metadata-thejoker and metadata-mcmc...Prepare-VAC.ipynb? Could make a script that joins the two into metadata.fits

* 7-null-control.sh
