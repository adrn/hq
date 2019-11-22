# Pipeline order:

* Create the run: `python make_run.py --name <your run name>`
* Edit the config file to set desired parameters
* Set the environment variable $HQ_RUN to `<your run name>` (or add
  `--name <your run name>` to all python calls below) to specify the HQ run name
* Create the prior cache: `python make_prior_cache.py`
* Set up the tasks used to parallelize and deploy: `python make_tasks.py`
* Run The Joker sampler on all stars: `python run_apogee.py`
* Analyze The Joker samplings to determine which stars are complete, which stars
  need to be followed up with standard MCMC:
  `python analyze_joker_samplings.py`
* Run standard MCMC on the unimodal samplings to generate 256 samples:
  `python run_continue_mcmc.py`




## Testing:

* `cd <hq repo>/scripts`
* `python make_run.py --name hqtest`
* `cp ../hq/pkgdata/_test_config.py ~/.hq/hqtest/config.py`
* `python make_prior_cache.py --name hqtest`
* `python make_tasks.py --name hqtest`
* `python run_apogee.py --name hqtest`
