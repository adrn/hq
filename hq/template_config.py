## This is a template configuration file for running HQ. Most of these settings
## are parameters that relate to The Joker: http://thejoker.readthedocs.io/
## For all possible parameters below, the parameter descriptions are preceded by
## "##" and the parameters (that can and should be edited) are preceded by "#"
## when optional, and are left unset when required (and will thus raise
## NameError's if not set)

## The name of the run (string):
# name = "hq-run"

#

name: {run_name}
requested_samples_per_star: 256 # M_min
hyperparams:
  P_min: 1 day
  P_max: 32768 day
  jitter_prior_mean: 9.5 # See infer-jitter notebook
  jitter_prior_stddev: 1.64
  jitter_prior_unit: m/s
  poly_trend: 2
prior:
  n_samples: 268435456
data:
    allstar: {allstar}
    allvisit: {allvisit}
    min_nvisits: 3
    # starflag_bits: [3, 4, 9, 12, 13, 16, 17]
    # aspcapflag_bits: [0, 8, 10, 23]
emcee:
    n_walkers: 512
    n_burn: 0
    n_steps: 8192
