"""
Given output from the initial run, finish sampling by running emcee (needs
mcmc).

This script must be run *after* run_apogee.py

TODO:
- For now, this script doesn't update the database. It just writes the chains
  out to files.

"""

# Standard library
from os import path
import os
import pickle

# Third-party
from astropy.stats import median_absolute_deviation
import h5py
import numpy as np
from schwimmbad import choose_pool
from thejoker.log import log as joker_logger
from thejoker.sampler import TheJoker
import yaml

# Project
from twoface.log import log as logger
from twoface.db import db_connect, get_run
from twoface.db import JokerRun, AllStar, StarResult, Status
from twoface.config import TWOFACE_CACHE_PATH
from twoface.mcmc_helpers import gelman_rubin, emcee_worker


def main(config_file, pool, seed, overwrite=False):
    config_file = path.abspath(path.expanduser(config_file))

    # parse config file
    with open(config_file, 'r') as f:
        config = yaml.load(f.read())

    # filename of sqlite database
    database_file = config['database_file']

    db_path = path.join(TWOFACE_CACHE_PATH, database_file)
    if not os.path.exists(db_path):
        raise IOError("sqlite database not found at '{0}'\n Did you run "
                      "scripts/initdb.py yet for that database?"
                      .format(db_path))

    logger.debug("Connecting to sqlite database at '{0}'".format(db_path))
    Session, engine = db_connect(database_path=db_path,
                                 ensure_db_exists=False)
    session = Session()

    run = get_run(config, session, overwrite=False)

    # The file with cached posterior samples:
    results_filename = path.join(TWOFACE_CACHE_PATH,
                                 "{0}.hdf5".format(run.name))
    if not path.exists(results_filename):
        raise IOError("Posterior samples result file {0} doesn't exist! Are "
                      "you sure you ran `run_apogee.py`?"
                      .format(results_filename))

    # The file to write the MCMC samples to:
    mcmc_filename = path.join(TWOFACE_CACHE_PATH,
                              "{0}-mcmc.hdf5".format(run.name))

    if not path.exists(mcmc_filename): # ensure it exists
        with h5py.File(mcmc_filename, 'w') as f:
            pass

    # Create TheJoker sampler instance with the specified random seed and pool
    rnd = np.random.RandomState(seed=seed)
    logger.debug("Creating TheJoker instance with {0}, {1}".format(rnd, pool))
    params = run.get_joker_params()

    # HACK: test
    params.jitter = (8.5, 0.9)
    joker = TheJoker(params, random_state=rnd)

    # Get all stars in this JokerRun that "need mcmc" that are not in the MCMC
    # cache file already
    with h5py.File(mcmc_filename, 'r') as f:
        done_ids = list(f.keys())

    base_query = session.query(AllStar).join(StarResult, JokerRun, Status)\
                                       .filter(JokerRun.name == run.name)\
                                       .filter(Status.id == 2)
    star_query = base_query.filter(~AllStar.apogee_id.in_(done_ids))

    n_stars = star_query.count()
    logger.info("{0} stars left to process for run more samples '{1}'"
                .format(n_stars, run.name))

    cache_path = path.join(TWOFACE_CACHE_PATH, 'emcee')
    logger.debug('Will write emcee chains to {0}'.format(cache_path))
    os.makedirs(cache_path, exist_ok=True)

    tasks = [(cache_path, results_filename, star.apogee_id,
              star.apogeervdata(), joker)
             for star in star_query.order_by(AllStar.apogee_id).all()]
    session.close()

    for r in pool.map(emcee_worker, tasks):
        pass

    pool.close()

    # Now go through all of the output files and collect them!
    with open(path.join(cache_path, 'model.pickle'), 'rb') as f:
        model = pickle.load(f)

    with h5py.File(mcmc_filename) as f:
        for star in base_query.all():
            if star.apogee_id in f:
                logger.debug('Star {0} already in MCMC cache file'
                             .format(star.apogee_id))
                continue

            tmp_file = path.join(cache_path, '{0}.npy'.format(star.apogee_id))
            chain = np.load(tmp_file)
            n_walkers, n_steps, n_pars = chain.shape

            g = f.create_group(star.apogee_id)

            logger.debug('Adding star {0} to MCMC cache'.format(star.apogee_id))
            try:
                g2 = g.create_group('chain-stats')

                # compute running median, MAD, mean, stddev
                all_med = []
                all_mad = []
                all_mean = []
                all_std = []
                for k in range(chain.shape[-1]):
                    all_med.append(np.median(chain[..., k], axis=0))
                    all_mad.append(median_absolute_deviation(chain[..., k],
                                                             axis=0))
                    all_mean.append(np.mean(chain[..., k], axis=0))
                    all_std.append(np.std(chain[..., k], axis=0))

                all_med = np.vstack(all_med).T
                all_mad = np.vstack(all_mad).T
                all_mean = np.vstack(all_mean).T
                all_std = np.vstack(all_std).T
                Rs = gelman_rubin(chain[:, n_steps//2:])

                g2.create_dataset(name='median', data=all_med)
                g2.create_dataset(name='MAD', data=all_mad)
                g2.create_dataset(name='mean', data=all_mean)
                g2.create_dataset(name='std', data=all_std)
                g2.create_dataset(name='gelman_rubin', data=Rs)

                # take the last sample, downsample
                end_pos = chain[:run.requested_samples_per_star, -1]
                samples = model.unpack_samples_mcmc(end_pos)
                samples.to_hdf5(g)

            except Exception as e:
                raise

            finally:
                del g


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

    parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
                        help="Random number seed")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("-c", "--config", dest="config_file", required=True,
                        type=str, help="Path to config file that specifies the "
                                       "parameters for this JokerRun.")

    args = parser.parse_args()

    loggers = [joker_logger, logger]

    # Set logger level based on verbose flags
    if args.verbosity != 0:
        if args.verbosity == 1:
            logger.setLevel(logging.DEBUG)
        else: # anything >= 2
            logger.setLevel(1)
            joker_logger.setLevel(1)

    elif args.quietness != 0:
        if args.quietness == 1:
            logger.setLevel(logging.WARNING)
            joker_logger.setLevel(logging.WARNING)
        else: # anything >= 2
            logger.setLevel(logging.ERROR)
            joker_logger.setLevel(logging.ERROR)

    else: # default
        logger.setLevel(logging.INFO)
        joker_logger.setLevel(logging.INFO)

    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)
    pool = choose_pool(**pool_kwargs)

    main(config_file=args.config_file, pool=pool, seed=args.seed)
