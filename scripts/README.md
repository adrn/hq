Order of operations:

* Create the run and fill template config file: `python make_run.py --name
  <run name> --allstar <path> --allvisit <path>`
* Edit the config file to set desired parameters
* Create the prior cache: `python make_prior_cache.py --name <run name>`
* Run The Joker sampler on all stars: `python run_apogee.py --name <run name>`
