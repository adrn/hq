# Standard library
from argparse import ArgumentParser
import logging
from os.path import abspath, expanduser, join
import os
import sys
import time

# Third-party
import astropy.units as u
import h5py
import numpy as np
from thejoker.log import log as joker_logger
from thejoker.sampler import TheJoker
from tqdm import tqdm
import yaml
from schwimmbad import SerialPool
from schwimmbad.mpi import MPIAsyncPool

# Project
from hq.log import log as logger
from hq.db import db_connect, get_run
from hq.db import JokerRun, AllStar, StarResult, Status
from hq.config import HQ_CACHE_PATH


##############################################################################
# TODO: shite
#
seed = 42
#
##############################################################################


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
if args.verbosity == 1:
    level = logging.DEBUG
elif args.verbosity == 2:
    level = 1
elif args.quietness == 1:
    level = logging.WARNING
elif args.quietness == 2:
    level = logging.ERROR
else:
    level = logging.INFO

for l in [logger, joker_logger]:
    l.setLevel(level)

if args.mpi:
    Pool = MPIAsyncPool
elif args.n_procs > 1:
    raise NotImplementedError("Sorry.")
else:
    Pool = SerialPool

# Load the prior samples in every process:
with open(args.config_file, 'r') as f:
    config = yaml.load(f.read())
prior_cache_file = join(HQ_CACHE_PATH, config['prior']['samples_file'])
n_prior_samples = config['prior']['max_samples']

# Create a file to cache the resulting posterior samples
results_filename = join(HQ_CACHE_PATH, "{0}.hdf5".format(config['name']))

with h5py.File(prior_cache_file, 'r') as f:
    prior_samples = np.array(f['samples']).astype('f8')
    prior_units = [u.Unit(uu) for uu in f.attrs['units']]
    ln_prior_probs = np.array(f['ln_prior_probs'])

n_requested_samples = config['hyperparams']['requested_samples_per_star']

# ----------------------------------------------------------------------------

def worker(joker, apogee_id, data):
    t0 = time.time()
    logger.log(1, "{0}: Starting sampling".format(apogee_id))

    try:
        samples, ln_prior, ln_likelihood = joker.iterative_rejection_sample(
            data=data, n_requested_samples=n_requested_samples,
            prior_samples=prior_samples, prior_units=prior_units,
            ln_prior_probs=ln_prior_probs, return_logprobs=True)
    except Exception as e:
        logger.warning("\t Failed sampling for star {0} \n Error: {1}"
                       .format(apogee_id, str(e)))
        return None

    logger.debug("{0}: done sampling - {1} raw samples returned "
                 "({2:.2f} seconds)".format(apogee_id, len(samples),
                                            time.time() - t0))

    return apogee_id, samples, ln_prior, ln_likelihood


def callback(future):
    res = future.result()

    if res is None:
        return

    apogee_id, samples, ln_prior, ln_likelihood = res

    # Ensure that the results file exists - this is where we cache samples that
    # pass the rejection sampling step
    with h5py.File(results_filename, 'a') as results_f:
        g = results_f.create_group(apogee_id)
        samples.to_hdf5(g)

        g.create_dataset('ln_prior', data=ln_prior)
        g.create_dataset('ln_likelihood', data=ln_likelihood)

    logger.debug("{0}: done, {1} samples returned ".format(apogee_id,
                                                           len(samples)))


def main(config_file, pool, overwrite=False):
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
    joker = TheJoker(params, random_state=rnd)
    logger.debug("Processing pool has size = {0}".format(pool.size))

    # Query to get all stars associated with this run:
    star_query = session.query(AllStar)\
                        .join(StarResult, JokerRun, Status)\
                        .filter(AllStar.apogee_id != '')\
                        .filter(JokerRun.name == run.name)\
                        .group_by(AllStar.apogee_id).distinct()

    n_stars = star_query.count()
    logger.info("{0} stars left to process for run '{1}'"
                .format(n_stars, run.name))

    tasks = []
    for star in star_query.all():
        # Write the samples that pass to the results file
        if star.apogee_id in results_f:
            if overwrite:
                del results_f[star.apogee_id]
            else:
                continue

        data = star.get_rvdata()
        tasks.append([joker, star.apogee_id, data])

    logger.info('{0} stars in process queue'.format(len(tasks)))

    for r in tqdm(pool.map(worker, tasks, callback=callback),
                  total=len(tasks)):
        pass


with Pool() as pool:
    main(config_file=args.config_file, pool=pool,
         overwrite=args.overwrite)

sys.exit(0)
