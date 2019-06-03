# Pipeline order:

* Create the run and fill template config file: `python make_run.py --name
  <run name> --allstar <path> --allvisit <path>`
* Edit the config file to set desired parameters
* Create the prior cache: `python make_prior_cache.py --name <run name>`
* Run The Joker sampler on all stars: `python run_apogee.py --name <run name>`
* Analyze The Joker samplings to determine which stars are complete, which stars
  need to be followed up with standard MCMC:
  `python analyze_joker_samplings.py --name <run name>`
* Run standard MCMC on the unimodal samplings to generate 256 samples:
  `python run_continue_mcmc.py --name <run name>`




## Testing:

* python make_run.py --name test --allstar ../hq/tests/test-allStar.fits --allvisit ../hq/tests/test-allVisit.fits
* python make_prior_cache.py --name test
* python run_apogee.py --name test
