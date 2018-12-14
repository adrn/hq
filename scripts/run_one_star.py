"""
Generate posterior samples over orbital parameters for a given APOGEE ID.

How to use
==========

TODO: pass in prior params via P_min, P_max, jitter

"""

# Standard library
from os.path import join
import os
import sys
import time

# Third-party
import astropy.units as u
import h5py
import numpy as np
from schwimmbad import choose_pool
from thejoker.log import log as joker_logger
from thejoker import TheJoker, JokerParams

# Project
from twoface.log import log as logger
from twoface.db import db_connect
from twoface.db import JokerRun, AllStar, StarResult, Status
from twoface.config import TWOFACE_CACHE_PATH
from twoface.sample_prior import make_prior_cache


def main(db_file, pool, seed, overwrite=False):

    db_path = join(TWOFACE_CACHE_PATH, db_file)
    if not os.path.exists(db_path):
        raise IOError("sqlite database not found at '{0}'\n Did you run "
                      "scripts/initdb.py yet for that database?"
                      .format(db_path))

    logger.debug("Connecting to sqlite database at '{0}'".format(db_path))
    Session, engine = db_connect(database_path=db_path,
                                 ensure_db_exists=False)
    session = Session()

    # TODO: all hard-set, these should be options
    params = JokerParams(P_min=10 * u.day,
                         P_max=1000 * u.day,
                         jitter=(9.5, 1.64),
                         jitter_unit=u.m/u.s)
    n_prior_samples = 2**28
    run_name = 'apogee-jitter'
    apogee_id = '2M01231070+1801407'

    results_filename = join(TWOFACE_CACHE_PATH, '{0}.hdf5'.format(apogee_id))
    prior_samples_file = join(TWOFACE_CACHE_PATH,
                              '{0}-prior.hdf5'.format(apogee_id))

    # Create TheJoker sampler instance with the specified random seed and pool
    rnd = np.random.RandomState(seed=seed)
    logger.debug("Creating TheJoker instance with {0}, {1}".format(rnd, pool))
    joker = TheJoker(params, random_state=rnd, pool=pool)

    # Create a cache of prior samples (if it doesn't exist) and store the
    # filename in the database.
    if not os.path.exists(prior_samples_file) or overwrite:
        logger.debug("Prior samples file not found - generating {0} samples..."
                     .format(n_prior_samples))
        make_prior_cache(prior_samples_file, joker,
                         nsamples=n_prior_samples)
        logger.debug("...done")

    # Query to get all stars associated with this run that need processing:
    # they should have a status id = 0 (needs processing)
    star_query = session.query(AllStar).join(StarResult, JokerRun, Status)\
                                       .filter(JokerRun.name == run_name)\
                                       .filter(AllStar.apogee_id == apogee_id)
    star = star_query.limit(1).one()

    logger.log(1, "Starting star {0}".format(star.apogee_id))
    t0 = time.time()

    data = star.apogeervdata()
    logger.log(1, "\t visits loaded ({:.2f} seconds)"
               .format(time.time()-t0))
    try:
        samples = joker.rejection_sample(
            data=data, prior_cache_file=prior_samples_file,
            return_logprobs=False)

    except Exception as e:
        logger.warning("\t Failed sampling for star {0} \n Error: {1}"
                       .format(star.apogee_id, str(e)))
        pool.close()
        sys.exit(1)

    logger.debug("\t done sampling ({:.2f} seconds)".format(time.time()-t0))

    # Write the samples that pass to the results file
    with h5py.File(results_filename, 'w') as f:
        samples.to_hdf5(f)

    logger.debug("\t saved samples ({:.2f} seconds)".format(time.time()-t0))
    logger.debug("...done with star {} ({:.2f} seconds)"
                 .format(star.apogee_id, time.time()-t0))

    pool.close()


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

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False, action='store_true')

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("--db", dest="db_file", required=True,
                        type=str, help="Path to the database file")

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

    main(db_file=args.db_file, pool=pool, seed=args.seed, overwrite=args.overwrite)
