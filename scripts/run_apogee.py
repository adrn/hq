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
import shutil

# Third-party
import h5py
import numpy as np
from schwimmbad import choose_pool
from thejoker.log import log as joker_logger
from thejoker.sampler import TheJoker, JokerSamples
import yaml

# Project
from hq.log import log as logger
from hq.db import db_connect, get_run
from hq.db import JokerRun, AllStar, StarResult, Status
from hq.config import HQ_CACHE_PATH
from hq.sample_prior import make_prior_cache
from hq.samples_analysis import unimodal_P


def cache_copy(prior_samples_file):
    from mpi4py import MPI
    rank = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', None)

    path, filename = os.path.split(prior_samples_file)
    dest = join('/dev/shm/', filename)

    if rank is None or str(rank).strip() != '1':
        logger.log(0, "Process {} on {} exiting"
                   .format(rank, MPI.Get_processor_name()))
        return dest

    logger.debug("Process {} on {} copying prior cache file..."
                 .format(rank, MPI.Get_processor_name()))
    if not os.path.exists(dest):
        shutil.copy(prior_samples_file, dest)
    return dest


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
    if not pool.size:
        n_batches = 128 # MAGIC NUMBER
    else:
        n_batches = 128 * pool.size

    logger.debug("Creating TheJoker instance with {0}, {1}".format(rnd, pool))
    joker = TheJoker(params, random_state=rnd, pool=pool)
    logger.debug("Processing pool has size = {0}".format(pool.size))

    # Create a cache of prior samples (if it doesn't exist) and store the
    # filename in the database.
    if not os.path.exists(run.prior_samples_file):
        logger.debug("Prior samples file not found - generating {0} samples..."
                     .format(config['prior']['num_cache']))
        make_prior_cache(run.prior_samples_file, joker,
                         nsamples=config['prior']['num_cache'])
        logger.debug("...done")
    else:
        logger.info("Using pre-generated prior sample cache at {0}. Delete this"
                    " if you'd like to re-generate!"
                    .format(run.prior_samples_file))

    # Get done APOGEE ID's
    done_subq = session.query(AllStar.apogee_id)\
                       .join(StarResult, JokerRun, Status)\
                       .filter(JokerRun.name == run.name)\
                       .filter(Status.id > 0).distinct()

    # Query to get all stars associated with this run that need processing:
    # they should have a status id = 0 (needs processing)
    star_query = session.query(AllStar)\
                        .join(StarResult, JokerRun, Status)\
                        .filter(AllStar.apogee_id != '')\
                        .filter(JokerRun.name == run.name)\
                        .filter(~AllStar.apogee_id.in_(done_subq))\
                        .filter(Status.id == 0)\
                        .group_by(AllStar.apogee_id).distinct()

    # Base query to get a StarResult for a given Star so we can update the
    # status, etc.
    result_query = session.query(StarResult).join(AllStar, JokerRun)\
                                            .filter(JokerRun.name == run.name)\
                                            .filter(Status.id == 0)\
                                            .filter(~AllStar.apogee_id.in_(done_subq))

    # Create a file to cache the resulting posterior samples
    results_filename = join(HQ_CACHE_PATH, "{0}.hdf5".format(run.name))
    n_stars = star_query.count()
    logger.info("{0} stars left to process for run '{1}'"
                .format(n_stars, run.name))

    # Ensure that the prior cache file exists on each node for faster reading:
    # TODO: don't do this on perseus
    if 'MPI' in pool.__class__.__name__:
        logger.debug("Copying prior cache file to nodes")
        for r in pool.map(cache_copy, [run.prior_samples_file] * pool.size):
            prior_cache_file_on_node = r
        logger.debug("...done copying prior cache file to nodes")
    else:
        prior_cache_file_on_node = run.prior_samples_file

    # Ensure that the results file exists - this is where we cache samples that
    # pass the rejection sampling step
    if not os.path.exists(results_filename):
        with h5py.File(results_filename, 'w') as f:
            pass

    results_f = h5py.File(results_filename, 'r+')

    # --------------------------------------------------------------------------
    # Here is where we do the actual processing of the data for each star. We
    # loop through all stars that still need processing and iteratively
    # rejection sample with larger and larger prior sample batch sizes. We do
    # this for efficiency, but the argument for this is somewhat made up...
    stars = star_query.all()

    count = 0 # how many stars we've processed in this star batch
    batch_size = 16 # MAGIC NUMBER: how many stars to process before committing
    for star in stars:

        if result_query.filter(AllStar.apogee_id == star.apogee_id).count() < 1:
            logger.debug('Star {0} has no result object!'
                         .format(star.apogee_id))
            continue

        # Retrieve existing StarResult from database. We limit(1) because the
        # APOGEE_ID isn't unique, but we attach all visits for a given star to
        # all rows, so grabbing one of them is fine.
        result = result_query.filter(AllStar.apogee_id == star.apogee_id)\
                             .limit(1).one()

        logger.log(1, "Starting star '{0}'".format(star.apogee_id))
        logger.log(1, "Current status: '{0}'".format(str(result.status)))
        t0 = time.time()

        data = star.get_rvdata()
        logger.log(1, "\t {0} visits loaded ({1:.2f} seconds)"
                   .format(len(data.rv), time.time()-t0))

        # TODO: add another overwrite flag to overwrite this bit too:
        if star.apogee_id in results_f:
            samples = JokerSamples.from_hdf5(results_f[star.apogee_id])
            n_actual_samples = len(samples)

        else:
            try:
                samples, ln_prior, ln_likelihood = joker.iterative_rejection_sample(
                    data=data, n_requested_samples=run.requested_samples_per_star,
                    prior_cache_file=prior_cache_file_on_node,
                    n_prior_samples=run.max_prior_samples, return_logprobs=True)

            except Exception as e:
                logger.warning("\t Failed sampling for star {0} \n Error: {1}"
                               .format(star.apogee_id, str(e)))
                continue

            logger.debug("\t done sampling - {0} raw samples returned "
                         "({1:.2f} seconds)".format(len(samples),
                                                    time.time()-t0))

            # For now, it's sufficient to write the run results to an HDF5 file
            n = run.requested_samples_per_star
            all_ln_probs = ln_prior[:n]
            samples = samples[:n]
            n_actual_samples = len(all_ln_probs)

            # Write the samples that pass to the results file
            if star.apogee_id in results_f:
                del results_f[star.apogee_id]

            g = results_f.create_group(star.apogee_id)
            samples.to_hdf5(g)

            if 'ln_prior_probs' in g:
                del g['ln_prior_probs']
            g.create_dataset('ln_prior_probs', data=all_ln_probs)
            results_f.flush()

        logger.debug("\t saved samples ({:.2f} seconds)".format(time.time()-t0))

        if n_actual_samples >= run.requested_samples_per_star:
            result.status_id = 4 # completed

        elif n_actual_samples == 1:
            # Only one sample was returned - this is probably unimodal, so this
            # star needs MCMC
            result.status_id = 2 # needs mcmc

        else:

            if unimodal_P(samples, data):
                # Multiple samples were returned, but they look unimodal
                result.status_id = 2 # needs mcmc

            else:
                # Multiple samples were returned, but not enough to satisfy the
                # number requested in the config file
                result.status_id = 1 # needs more samples

        logger.debug("...done with star {0}: {1} visits, {2} samples returned "
                     " ({3:.2f} seconds)"
                     .format(star.apogee_id, len(data.rv), len(samples),
                             time.time()-t0))

        if count % batch_size == 0 and count > 0:
            session.commit()

        count += 1

    pool.close()

    session.commit()
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

    parser.add_argument("--overwrite", dest="overwrite", default=False,
                        action="store_true",
                        help="Overwrite any existing results for this "
                             "JokerRun.")

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

    main(config_file=args.config_file, pool=pool, seed=args.seed,
         overwrite=args.overwrite)
