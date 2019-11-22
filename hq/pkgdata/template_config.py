# flake8: noqa

## This is a template configuration file for running HQ. Most of these settings
## are parameters that relate to The Joker: http://thejoker.readthedocs.io/
## When settings are required, they are noted, and will error if left
## unmodified. Other parameters have defaults set in the file below.

## Imports we always need:
import astropy.units as u
import pymc3 as pm
import exoplanet.units as xu
import thejoker as tj

##############################################################################
## General parameters and data:
##

## The name of the run (string):
## **REQUIRED**: please edit this
name = None

## The paths to the APOGEE allStar and allVisit files:
## **REQUIRED**: please edit these
allstar_filename = None
allvisit_filename = None

## The minimum number of visits to accept when filtering sources:
min_nvisits = 3

##############################################################################
## Prior and sampling parameters for The Joker:
##

## The prior used to run The Joker:
## **REQUIRED**: please edit below
with pm.Model() as model:

  # Prior on the extra variance parameter:
  s = xu.with_unit(pm.Lognormal('s', 0, 0.5),
                   u.km/u.s)

  # Set up the default Joker prior:
  prior = tj.JokerPrior.default(
    P_min=None, P_max=None,
    sigma_K0=None,
    sigma_v=None
  )

## Number of prior samples to generate and cache:
n_prior_samples = 500_000_000

## Name of the prior cache file:
prior_cache_file = f'prior_samples_{n_samples}_{name}.hdf5'

## The number of posterior samples to generate per source:
requested_samples_per_star = 1024


##############################################################################
## Sampling parameters for MCMC:
##

## These parameters are passed directly to pymc3.sample()
tune = 1000
draws = 1000

## This is the target acceptance ratio used for computing the dense mass matrix:
target_accept = 0.95
