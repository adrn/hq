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

## Name of the prior definition file:
prior_file = 'prior.py'

## Number of prior samples to generate and cache:
n_prior_samples = 500_000_000

## Name of the prior cache file:
prior_cache_file = f'prior_samples_{n_prior_samples}_{name}.hdf5'

## The number of posterior samples to generate per source:
requested_samples_per_star = 1024

## Randomly draw samples from the prior cache. This will slow things down!
randomize_prior_order = False


##############################################################################
## Sampling parameters for MCMC:
##

## These parameters are passed directly to pymc3.sample()
tune = 1000
draws = 1000

## This is the target acceptance ratio used for computing the dense mass matrix:
target_accept = 0.95
