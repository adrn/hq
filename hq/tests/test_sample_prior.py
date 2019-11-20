# Third-party
import astropy.units as u
import h5py
import thejoker as tj

# Project
from ..sample_prior import make_prior_cache


def test_make_prior_cache(tmpdir):
    N = 2**18

    filename = str(tmpdir / 'prior_samples.h5')
    prior = tj.JokerPrior.default(
        P_min=8*u.day, P_max=8192*u.day,
        sigma_K0=30*u.km/u.s, sigma_v=100*u.km/u.s)

    make_prior_cache(filename, prior, nsamples=N, batch_size=2**14)

    with h5py.File(filename, 'r') as f:
        assert f['samples'].shape == (N, 5)
