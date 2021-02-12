# A pipeline tool for running The Joker on large datasets

[The Joker](https://github.com/adrn/thejoker) is a custom Monte Carlo sampler
for generating posterior samplings over Keplerian orbital parameters for
two-body systems. This package provides a pipeline for running The Joker on
large datasets of homogenous radial velocity data. Early versions of this tool
were used to analyze data from the [APOGEE
surveys](https://www.sdss.org/surveys/apogee-2/), leading to the discovery of
many thousands of binary star systems ([Price-Whelan et al.
2018](https://ui.adsabs.harvard.edu/abs/2018AJ....156...18P/abstract),
[Price-Whelan et al.
2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...895....2P/abstract)).

## Pipeline order

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


## License

This project is Copyright (c) Adrian Price-Whelan and licensed under the terms
of the MIT license. This package is based upon the [Astropy package
template](https://github.com/astropy/package-template) which is licensed under
the BSD 3-clause license.
