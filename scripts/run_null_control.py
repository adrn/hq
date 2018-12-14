"""
Take 10,000 APOGEE stars (1/10 of the full sample) and re-generate visit
velocities assuming that there are no true orbital RV variations.
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
from twoface.log import log as logger
from twoface.data import APOGEERVData
from twoface.db import db_connect, get_run
from twoface.db import JokerRun, AllStar, StarResult, Status
from twoface.config import TWOFACE_CACHE_PATH

# MAGIC NUMBER: number of control stars to test
NCONTROL = 16384


def main(config_file, pool, seed, overwrite=False):
    # Default seed:
    if seed is None:
        seed = 42

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

    db_path = join(TWOFACE_CACHE_PATH, database_file)
    if not os.path.exists(db_path):
        raise IOError("sqlite database not found at '{0}'\n Did you run "
                      "scripts/initdb.py yet for that database?"
                      .format(db_path))

    logger.debug("Connecting to sqlite database at '{0}'".format(db_path))
    Session, engine = db_connect(database_path=db_path,
                                 ensure_db_exists=False)
    session = Session()

    # Retrieve or create a JokerRun instance
    run = get_run(config, session, overwrite=False) # never overwrite
    params = run.get_joker_params()

    # Create TheJoker sampler instance with the specified random seed and pool
    rnd = np.random.RandomState(seed=seed)
    logger.debug("Creating TheJoker instance with {0}, {1}".format(rnd, pool))
    joker = TheJoker(params, random_state=rnd, pool=pool)

    # Create a file to cache the resulting posterior samples
    results_filename = join(TWOFACE_CACHE_PATH,
                            "{0}-control.hdf5".format(run.name))

    # Ensure that the results file exists - this is where we cache samples that
    # pass the rejection sampling step
    if not os.path.exists(results_filename):
        with h5py.File(results_filename, 'w') as f:
            pass

    with h5py.File(results_filename, 'r') as f:
        done_apogee_ids = list(f.keys())

    # Create a cache of prior samples (if it doesn't exist) and store the
    # filename in the database.
    if not os.path.exists(run.prior_samples_file):
        raise IOError("Prior cache must already exist.")

    # Get random IDs
    star_ids = session.query(AllStar.apogee_id)\
                      .join(StarResult, JokerRun, Status)\
                      .filter(Status.id > 0).distinct().all()
    star_ids = np.array([x[0] for x in star_ids])
    star_ids = rnd.choice(star_ids, size=NCONTROL, replace=False)
    star_ids = star_ids[~np.isin(star_ids, done_apogee_ids)]

    n_stars = len(star_ids)
    logger.info("{0} stars left to process for run '{1}'; {2} already done."
                .format(n_stars, run.name, len(done_apogee_ids)))

    # --------------------------------------------------------------------------
    # Here is where we do the actual processing of the data for each star. We
    # loop through all stars that still need processing and iteratively
    # rejection sample with larger and larger prior sample batch sizes. We do
    # this for efficiency, but the argument for this is somewhat made up...

    for apid in star_ids:
        star = AllStar.get_apogee_id(session, apid)

        logger.log(1, "Starting star {0}".format(star.apogee_id))
        t0 = time.time()

        orig_data = star.apogeervdata()

        # HACK: this assumes we're sampling over the excess variance parameter
        # Generate new data with no RV orbital variations
        y = rnd.normal(params.jitter[0], params.jitter[1])
        s = np.exp(0.5 * y) * params._jitter_unit
        std = np.sqrt(s**2 + orig_data.stddev**2).to(orig_data.rv.unit).value
        new_rv = rnd.normal(np.mean(orig_data.rv).value, std)
        data = APOGEERVData(t=orig_data.t, rv=new_rv * orig_data.rv.unit,
                            stddev=orig_data.stddev)

        logger.log(1, "\t visits loaded ({:.2f} seconds)"
                   .format(time.time()-t0))
        try:
            samples, ln_prior = joker.iterative_rejection_sample(
                data=data, n_requested_samples=run.requested_samples_per_star,
                prior_cache_file=run.prior_samples_file,
                n_prior_samples=run.max_prior_samples, return_logprobs=True)

        except Exception as e:
            logger.warning("\t Failed sampling for star {0} \n Error: {1}"
                           .format(star.apogee_id, str(e)))
            continue

        logger.debug("\t done sampling ({:.2f} seconds)".format(time.time()-t0))

        # For now, it's sufficient to write the run results to an HDF5 file
        n = run.requested_samples_per_star
        samples = samples[:n]

        # Write the samples that pass to the results file
        with h5py.File(results_filename, 'r+') as f:
            if star.apogee_id in f:
                del f[star.apogee_id]

            # HACK: this will overwrite the past samples!
            g = f.create_group(star.apogee_id)
            samples.to_hdf5(g)

        logger.debug("\t saved samples ({:.2f} seconds)".format(time.time()-t0))

    pool.close()
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

    oc_group = parser.add_mutually_exclusive_group()
    oc_group.add_argument("--overwrite", dest="overwrite", default=False,
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
