# Standard library
import sys
import time

# Third-party
import astropy.units as u
from astropy.time import Time
import numpy as np
from schwimmbad import choose_pool
import emcee
import matplotlib.pyplot as plt

from twobody import KeplerOrbit
from thejoker import RVData, JokerParams, JokerSamples
from thejoker.sampler.mcmc import TheJokerMCMCModel


def test_ln_prob(*args, **kwargs):
    time.sleep(0.5E-3)
    return -0.5


def make_model_ln_prob(p, pars, data):
    model = TheJokerMCMCModel(joker_params=pars, data=data)
    return model(p)


def main(pool_kwargs):

    # Make some fake data
    orbit = KeplerOrbit(P=30*u.day, e=0.1, M0=0*u.rad, omega=0*u.deg)
    K = 1*u.km/u.s
    samples0 = JokerSamples()
    samples0['P'] = 30*u.day
    samples0['e'] = 0.1*u.one
    samples0['M0'] = 0*u.rad
    samples0['omega'] = 0*u.rad
    samples0['jitter'] = 1*u.m/u.s
    samples0['K'] = 1*u.km/u.s
    samples0['v0'] = 0*u.km/u.s

    t = Time(59000 + np.linspace(0, 100, 16), format='mjd')
    err = np.ones_like(t.mjd) * 10*u.m/u.s
    data = RVData(t=t, rv=K * orbit.unscaled_radial_velocity(t),
                  stddev=err)

    params = JokerParams(P_min=1*u.day, P_max=1024*u.day)

    pool = choose_pool(**pool_kwargs)

    model = TheJokerMCMCModel(joker_params=params, data=data)

    p0_mean = np.squeeze(model.pack_samples(samples0))
    ball_std = 1E-3 # TODO: make this customizable?

    # P, M0, e, omega, jitter, K, v0
    n_walkers = 1024
    p0 = np.zeros((n_walkers, len(p0_mean)))
    for i in range(p0.shape[1]):
        if i in [2, 4]: # eccentricity, jitter
            p0[:, i] = np.abs(np.random.normal(p0_mean[i], ball_std,
                                               size=n_walkers))

        else:
            p0[:, i] = np.random.normal(p0_mean[i], ball_std,
                                        size=n_walkers)

    p0 = model.to_mcmc_params(p0.T).T

    # Because jitter is always carried through in the transform above, now
    # we have to remove the jitter parameter if it's fixed!
    if params._fixed_jitter:
        p0 = np.delete(p0, 5, axis=1)

    # time0 = time.time()
    # results = list(pool.map(model,
    # # results = list(pool.map(test_ln_prob,
    #                (p0[i] for i in range(len(p0)))))
    # print('...time spent mapping: {0}'.format(time.time()-time0))
    #
    # pool.close()
    # sys.exit(0)

    n_dim = p0.shape[1]
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, make_model_ln_prob,
                                    pool=pool, args=(params, data))

    n_burn = 128
    print('Burning in MCMC for {0} steps...'.format(n_burn))
    time0 = time.time()
    pos, *_ = sampler.run_mcmc(p0, n_burn)
    print('...time spent burn-in: {0}'.format(time.time()-time0))

    pool.close()
    sys.exit(0)


if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    args = parser.parse_args()

    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)
    main(pool_kwargs=pool_kwargs)
