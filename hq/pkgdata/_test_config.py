# flake8: noqa

## This is a configuration file used for internal HQ tests.

##############################################################################
## General parameters and data:
##

## The name of the run (string):
## **REQUIRED**: please edit this
name = 'hqtest'

## The paths to the APOGEE allStar and allVisit files:
from astropy.utils.data import get_pkg_data_filename
allstar_filename = get_pkg_data_filename('tests/test-allStar.fits',
                                         package='hq')
allvisit_filename = get_pkg_data_filename('tests/test-allVisit.fits',
                                          package='hq')

## The minimum number of visits to accept when filtering sources:
min_nvisits = 3

##############################################################################
## Prior and sampling parameters for The Joker:
##

## Name of the prior definition file:
prior_file = 'prior.py'

## Number of prior samples to generate and cache:
n_prior_samples = 1_000_000

## Name of the prior cache file:
prior_cache_file = f'prior_samples_{n_prior_samples}_{name}.hdf5'

## The number of posterior samples to generate per source:
requested_samples_per_star = 16


##############################################################################
## Sampling parameters for MCMC:
##

## These parameters are passed directly to pymc3.sample()
tune = 100
draws = 100

## This is the target acceptance ratio used for computing the dense mass matrix:
target_accept = 0.95
