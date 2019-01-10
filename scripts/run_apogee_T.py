"""
Generate posterior samples over orbital parameters for all stars in the
specified database file.

How to use
==========

This script is configured with a YAML configuration file that specifies the
parameters of the processing run. These are mainly hyper-parameters for
`The Joker <thejoker.readthedocs.io>`_, but  also specify things like the name
of the database file to pull data from, the name of the run, and the number of
prior samples to generate and cache. To see an example, check out the YAML file
at ``twoface/run_config/apogee.yml``.

Parallel processing
===================

This script is designed to be used on any machine, and supports parallel
processing on computers from laptop (via multiprocessing) to compute cluster
(via MPI). The type of parallelization is specified using the command-line
flags ``--mpi`` and ``--ncores``. By default (with no flags), all calculations
are done in serial.


"""

# Standard library
from os.path import abspath, expanduser, join
import os
import time

# Third-party
import h5py
import numpy as np
from schwimmbad import choose_pool
from thejoker.log import log as joker_logger
from thejoker.sampler import TheJoker
import yaml

# Project
from hq.log import log as logger
from hq.db import db_connect, get_run
from hq.db import JokerRun, AllStar, StarResult, Status
from hq.db.helpers import paged_query
from hq.config import HQ_CACHE_PATH
from hq.sample_prior import make_prior_cache
from hq.samples_analysis import unimodal_P


def worker(task):
    run, joker, star, results_file = task

    logger.log(1, "Starting star '{0}'".format(star.apogee_id))
    t0 = time.time()

    data = star.get_rvdata(clean=False)
    logger.log(1, "\t {0} visits loaded ({1:.2f} seconds)"
               .format(len(data.rv), time.time()-t0))
    try:
        samples, ln_prior_probs = joker.iterative_rejection_sample(
            data=data,
            n_requested_samples=run.requested_samples_per_star,
            prior_cache_file=run.prior_samples_file,
            n_prior_samples=run.max_prior_samples,
            return_logprobs=True)

    except Exception as e:
        logger.warning("\t Failed sampling for star {0} \n Error: {1}"
                       .format(star.apogee_id, str(e)))
        return None

    logger.debug("\t done sampling {0}: {1} visits, {2} samples returned "
                 "({3:.2f} seconds)".format(star.apogee_id, len(data.rv),
                                            len(samples), time.time()-t0))

    return star, samples, ln_prior_probs, results_file


def callback(result):
    if result is None:
        return

    star, samples, ln_prior_probs, results_file = result

    # Write the samples that pass to the results file
    with h5py.File(results_file, 'r+') as f:
        if star.apogee_id in f:
            del f[star.apogee_id]

        # HACK: this will overwrite the past samples!
        g = f.create_group(star.apogee_id)
        samples.to_hdf5(g)

        if 'ln_prior_probs' in g:
            del g['ln_prior_probs']
        g.create_dataset('ln_prior_probs', data=ln_prior_probs)

    logger.log(1, "\t saved samples")


def main(config_file, pool, seed, overwrite=False):
    config_file = abspath(expanduser(config_file))

    # parse config file
    with open(config_file, 'r') as f:
        config = yaml.load(f.read())
        config['config_file'] = config_file

    # filename of sqlite database
    if 'database_file' not in config:
        database_file = None

    else:
        database_file = config['database_file']

    db_path = join(HQ_CACHE_PATH, database_file)
    if not os.path.exists(db_path):
        raise IOError("sqlite database not found at '{0}'\n Did you run "
                      "scripts/initdb.py yet for that database?"
                      .format(db_path))

    logger.debug("Connecting to sqlite database at '{0}'".format(db_path))
    Session, engine = db_connect(database_path=db_path,
                                 ensure_db_exists=False)
    session = Session()

    # Retrieve or create a JokerRun instance
    run = get_run(config, session, overwrite=overwrite)
    params = run.get_joker_params()

    # Create TheJoker sampler instance with the specified random seed and pool
    rnd = np.random.RandomState(seed=seed)
    logger.debug("Creating TheJoker instance with {0}".format(rnd))
    joker = TheJoker(params, random_state=rnd, n_batches=16) # MAGIC NUMBER

    # Create a cache of prior samples (if it doesn't exist) and store the
    # filename in the database.
    if not os.path.exists(run.prior_samples_file) or overwrite:
        logger.debug("Prior samples file not found - generating {0} samples..."
                     .format(config['prior']['num_cache']))
        make_prior_cache(run.prior_samples_file, joker,
                         nsamples=config['prior']['num_cache'])
        logger.debug("...done")

    # Get done APOGEE ID's
    done_subq = session.query(AllStar.apogee_id)\
                       .join(StarResult, JokerRun, Status)\
                       .filter(Status.id > 0).distinct()

    # Query to get all stars associated with this run that need processing:
    # they should have a status id = 0 (needs processing)
    star_query = session.query(AllStar)\
                        .join(StarResult, JokerRun, Status)\
                        .filter(JokerRun.name == run.name)\
                        .filter(AllStar.apogee_id != "")\
                        .filter(Status.id == 0)\
                        .filter(~AllStar.apogee_id.in_(done_subq))\
                        .order_by(AllStar.apogee_id).distinct()

    # Create a file to cache the resulting posterior samples
    results_filename = join(HQ_CACHE_PATH, "{0}.hdf5".format(run.name))
    n_stars = star_query.count()
    logger.info("{0} stars left to process for run '{1}'"
                .format(n_stars, run.name))

    # Ensure that the results file exists - this is where we cache samples that
    # pass the rejection sampling step
    if not os.path.exists(results_filename):
        with h5py.File(results_filename, 'w') as f:
            pass

    # --------------------------------------------------------------------------
    # Here is where we do the actual processing of the data for each star. We
    # loop through all stars that still need processing and iteratively
    # rejection sample with larger and larger prior sample batch sizes. We do
    # this for efficiency, but the argument for this is somewhat made up...

    # batch_size = 1024
    batch_size = 10
    for i, sub_q in enumerate(paged_query(star_query, page_size=batch_size)):
        stars = sub_q.all()
        tasks = [(run, joker, s, results_filename) for s in stars]
        logger.debug("Running batch {0}, {1} stars".format(i, len(tasks)))
        for r in pool.map(worker, tasks, callback=callback):
            pass

        break

    pool.close()

    # session.commit()
    session.close()


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
