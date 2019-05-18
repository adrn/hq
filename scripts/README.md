# Pipeline order:

* Create the run and fill template config file: `python make_run.py --name
  <run name> --allstar <path> --allvisit <path>`
* Edit the config file to set desired parameters
* Create the prior cache: `python make_prior_cache.py --name <run name>`
* Run The Joker sampler on all stars: `python run_apogee.py --name <run name>`


## Testing:

* python make_run.py --name test --allstar ../hq/tests/test-allStar.fits --allvisit ../hq/tests/test-allVisit.fits
* python make_prior_cache.py --name test
* python run_apogee.py --name test
