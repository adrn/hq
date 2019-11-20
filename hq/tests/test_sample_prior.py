import logging

# Third-party
import astropy.units as u
import h5py
import thejoker as tj

# Project
from ..sample_prior import make_prior_cache
from ..log import logger


def test_make_prior_cache(tmpdir):
    N = 2**18

    filename = str(tmpdir / 'prior_samples.h5')
    prior = tj.JokerPrior.default(
        P_min=8*u.day, P_max=8192*u.day,
        sigma_K0=30*u.km/u.s, sigma_v=100*u.km/u.s)

    old_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    make_prior_cache(filename, prior, n_total_samples=N, batch_size=2**16)
    logger.setLevel(old_level)

    with h5py.File(filename, 'r') as f:
        assert f['samples'].shape == (N, )
