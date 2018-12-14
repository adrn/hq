# Standard library
import os
from os import path
import sys

# Third-party
from astropy.time import Time
import astropy.units as u
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import schwimmbad

from twobody import KeplerOrbit

from thejoker import RVData, JokerParams, TheJoker
from thejoker.log import log as joker_logger
from thejoker.sampler import pack_prior_samples

from twoface.log import log as logger


def random_orbit(P, circ=False):
    kw = dict()
    # kw['P'] = 2 ** np.random.uniform(2, 10) * u.day # 4 to 1024 days
    # kw['P'] = 128. * u.day # MAGIC NUMBER
    kw['P'] = P * u.day
    kw['M0'] = np.random.uniform(0, 2*np.pi) * u.rad
    kw['e'] = 0.
    kw['omega'] = 0 * u.deg

    if not circ:
        kw['e'] = 10 ** np.random.uniform(-4, -0.001)
        kw['omega'] = np.random.uniform(0, 2*np.pi) * u.rad
        kw['Omega'] = np.random.uniform(0, 2*np.pi) * u.rad
        kw['i'] = np.arccos(np.random.uniform(0, 1)) * u.rad

    return KeplerOrbit(**kw)


def make_data(n_epochs, P, n_orbits=128, time_sampling='uniform', circ=False):
    """
    time_sampling can be 'uniform' or 'log'
    """

    # Hard-set MAGIC NUMBERs:
    t0 = Time('2013-01-01')
    baseline = 5 * u.yr # similar to APOGEE2
    K = 1 * u.km/u.s
    err = 150 * u.m/u.s

    if time_sampling == 'uniform':
        def t_func(size):
            return np.random.uniform(t0.mjd, (t0 + baseline).mjd, size=size)

    elif time_sampling == 'log':
        def t_func(size):
            t1 = np.log10(baseline.to(u.day).value)
            t = t0 + 10 ** np.random.uniform(0, t1, size=size) * u.day
            return t

    else:
        raise ValueError('invalid time_sampling value')

    for N in n_epochs:
        for i in range(n_orbits):
            orb = random_orbit(P=P, circ=circ)
            t = Time(t_func(N), format='mjd')
            rv = K * orb.unscaled_radial_velocity(t)
            rv = np.random.normal(rv.to(u.km/u.s).value,
                                  err.to(u.km/u.s).value) * u.km/u.s
            data = RVData(t, rv, stddev=np.ones_like(rv.value) * err)
            yield N, i, data, orb.P.to(u.day).value


def make_caches(N, joker, time_sampling, P, circ=False, overwrite=False):
    if not path.exists('exp2.py'):
        raise RuntimeError('This script must be run from inside '
                           'scripts/exp2-num-epochs')

    cache_path = path.abspath(path.join(os.getcwd(), 'cache'))
    if not path.exists(cache_path):
        os.makedirs(cache_path)

    if circ:
        prior_filename = path.join(cache_path, 'circ-prior-samples.hdf5')
        post_filename = path.join(cache_path, 'circ-samples-{0}-{1}.hdf5')
    else:
        prior_filename = path.join(cache_path, 'ecc-prior-samples.hdf5')
        post_filename = path.join(cache_path, 'ecc-samples-{0}-{1}.hdf5')

    post_filename = post_filename.format(time_sampling, int(P))

    if not path.exists(prior_filename) and not overwrite:
        samples, ln_probs = joker.sample_prior(N, return_logprobs=True)
        packed_samples, units = pack_prior_samples(samples, u.km/u.s)

        if circ:
            packed_samples[:, 2] = 0. # eccentricity
            packed_samples[:, 3] = 0. # omega

        with h5py.File(prior_filename, 'w') as f:
            # make the HDF5 file with placeholder datasets
            f.create_dataset('samples', data=packed_samples)
            f.attrs['units'] = np.array([str(x)
                                         for x in units]).astype('|S6')

    if not path.exists(post_filename) and not overwrite:
        # ensure prior cache file exists
        with h5py.File(post_filename, 'w') as f:
            f.attrs['P'] = P

    return prior_filename, post_filename


def main(pool, circ, P, time_sampling, overwrite=False):

    pars = JokerParams(P_min=1*u.day, P_max=1024*u.day)
    joker = TheJoker(pars, pool=pool)

    # make the prior cache file
    prior_file, samples_file = make_caches(2**28, joker, circ=circ, P=P,
                                           overwrite=overwrite,
                                           time_sampling=time_sampling)

    logger.info('Files: {0}, {1}'.format(prior_file, samples_file))

    n_epochs = np.arange(3, 12+1, 1)
    for n_epoch, i, data, P in make_data(n_epochs, n_orbits=512, P=P,
                                         time_sampling=time_sampling,
                                         circ=circ):
        logger.debug("N epochs: {0}, orbit {1}".format(n_epoch, i))

        key = '{0}-{1}'.format(n_epoch, i)
        with h5py.File(samples_file, 'r') as f:
            if key in f:
                logger.debug('-- already done! skipping...')
                continue

        samples = joker.iterative_rejection_sample(n_requested_samples=256,
                                                   prior_cache_file=prior_file,
                                                   data=data)
        logger.debug("-- done sampling - {0} samples returned"
                     .format(len(samples)))

        with h5py.File(samples_file) as f:
            g = f.create_group(key)
            samples.to_hdf5(g)
            g.attrs['P'] = P


if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0,
                          dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0,
                          dest='quietness')

    # multiprocessing
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    # Note: default seed is set!
    parser.add_argument('-s', '--seed', dest='seed', default=42,
                        type=int, help='Random number generator seed.')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        dest='overwrite', default=False,
                        help='Destroy everything.')

    parser.add_argument('--circ', action='store_true',
                        dest='circ', default=False)
    parser.add_argument('-P', default=128, type=int,
                        dest='P')
    parser.add_argument('--time_sampling', type=str, default='uniform',
                        dest='time_sampling')

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbosity != 0:
        if args.verbosity == 1:
            logger.setLevel(logging.DEBUG)
        else: # anything >= 2
            logger.setLevel(1)
            joker_logger.setLevel(logging.DEBUG)

    elif args.quietness != 0:
        if args.quietness == 1:
            logger.setLevel(logging.WARNING)
        else: # anything >= 2
            logger.setLevel(logging.ERROR)
            joker_logger.setLevel(logging.ERROR)

    else: # default
        logger.setLevel(logging.INFO)

    if args.seed is not None:
        np.random.seed(args.seed)

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)

    try:
        main(pool=pool, circ=args.circ, P=args.P,
             time_sampling=args.time_sampling, overwrite=args.overwrite)

    except Exception as e:
        raise e

    finally:
        pool.close()
