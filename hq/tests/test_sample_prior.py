# Third-party
import astropy.units as u
import h5py
from thejoker.sampler import TheJoker, JokerParams

# Project
from ..sample_prior import make_prior_cache

def test_make_prior_cache(tmpdir):
    N = 2**18

    filename = str(tmpdir / 'prior_samples.h5')
    params = JokerParams(P_min=8*u.day, P_max=8192*u.day)
    joker = TheJoker(params)

    make_prior_cache(filename, joker, nsamples=N, batch_size=2**14)

    with h5py.File(filename, 'r') as f:
        assert f['samples'].shape == (N, 5)
