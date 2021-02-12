A pipeline for running The Joker on APOGEE data
-----------------------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge


License
-------

This project is Copyright (c) Adrian Price-Whelan and licensed under the terms
of the MIT license. This package is based upon the `Astropy package
template <https://github.com/astropy/package-template>`_ which is licensed under
the BSD 3-clause license.


Pipeline order
--------------

* Create the run: `hq init --path <path for run>`
* Edit the config file to set desired parameters
* Set the environment variable $HQ_RUN_PATH to `<path for run>` (or add
  `--path <path for run>` to all python calls below) to specify the HQ run
* Create the prior cache: `hq make_prior_cache`
* Set up the tasks used to parallelize and deploy: `hq make_tasks`
* Run The Joker sampler on all stars: `hq run_thejoker`
* (optional) Fit the robust constant RV model to all sources: `hq run_constant`
* Analyze The Joker samplings to determine which stars are complete, which stars
  need to be followed up with standard MCMC:
  `hq analyze_joker`
* Run standard MCMC on the unimodal samplings: `hq run_mcmc`
* Analyze the MCMC samplings: `hq analyze_mcmc`
* Combine the metadata files: `hq combine_metadata`


Testing
-------

Old / out of date...

* `python init_run.py --name hqtest`
* `cp ../hq/pkgdata/_test_config.yml $HQ_CACHE_PATH/hqtest/config.yml`
* `cp ../hq/pkgdata/_test_prior.py $HQ_CACHE_PATH/hqtest/prior.py`
* `python make_prior_cache.py --name hqtest`
* `python make_tasks.py --name hqtest -o`
* `mpirun -n 4 python run_apogee.py --name hqtest -o --mpi -v`
* `python run_fit_constant.py --name hqtest -o`
* `python analyze_joker_samplings.py --name hqtest -o -v`
* `python run_continue_mcmc.py --name hqtest -o -v` (to get the number of rows)
* `python run_continue_mcmc.py --name hqtest --num 0 -o -v`
* `python analyze_mcmc_samplings.py --name hqtest -o -v`
* `python run_null_control.py --name hqtest -v`