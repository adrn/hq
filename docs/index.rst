Documentation
=============

This is the documentation for hq.


.. toctree::
  :maxdepth: 2

  hq/index.rst

.. note:: The layout of this directory is simply a suggestion.  To follow
          traditional practice, do *not* edit this page, but instead place
          all documentation for the package inside ``hq/``.
          You can follow this practice or choose your own layout.

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
*