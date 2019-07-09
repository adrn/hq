# Standard library
from os.path import join, exists
import sys
import time

# Third-party
import h5py
import numpy as np
from thejoker.log import log as joker_logger
from thejoker.sampler import TheJoker
from tqdm import tqdm
import yaml
from schwimmbad import SerialPool
from schwimmbad.mpi import MPIAsyncPool
from thejoker.data import RVData

# Project
from hq.log import logger
from hq.config import (HQ_CACHE_PATH, config_to_jokerparams,
                       config_to_prior_cache, config_to_alldata)
from hq.script_helpers import get_parser


def worker(joker, apogee_id, data, config, results_filename):
    t0 = time.time()
    logger.log(1, "{0}: Starting sampling".format(apogee_id))

    prior_cache_file = config_to_prior_cache(config, joker.params)
    try:
        samples, ln_prior, ln_likelihood = joker.iterative_rejection_sample(
            data=data, n_requested_samples=config['requested_samples_per_star'],
            prior_cache_file=prior_cache_file, return_logprobs=True)
    except Exception as e:
        logger.warning("\t Failed sampling for star {0} \n Error: {1}"
                       .format(apogee_id, str(e)))
        return None

    logger.debug("{0}: done sampling - {1} raw samples returned "
                 "({2:.2f} seconds)".format(apogee_id, len(samples),
                                            time.time() - t0))

    res = dict()
    res['apogee_id'] = apogee_id
    res['samples'] = samples
    res['ln_prior'] = ln_prior
    res['ln_likelihood'] = ln_likelihood
    res['results_filename'] = results_filename
    return res


def callback(future):
    if isinstance(future, dict) or future is None:
        res = future
    else:
        res = future.result()

    if res is None:
        return

    # Ensure that the results file exists - this is where we cache samples that
    # pass the rejection sampling step
    with h5py.File(res['results_filename'], 'a') as results_f:
        if res['apogee_id'] in results_f:  # TODO: is this ok?
            del results_f[res['apogee_id']]

        g = results_f.create_group(res['apogee_id'])
        res['samples'].to_hdf5(g)

        g.create_dataset('ln_prior', data=res['ln_prior'])
        g.create_dataset('ln_likelihood', data=res['ln_likelihood'])

    logger.debug("{0}: done, {1} samples returned ".format(res['apogee_id'],
                                                           len(res['samples'])))


def main(run_name, pool, overwrite=False, seed=None):
    run_path = join(HQ_CACHE_PATH, run_name)
    with open(join(run_path, 'config.yml'), 'r') as f:
        config = yaml.load(f.read())

    # Get paths to files needed to run
    params = config_to_jokerparams(config)
    prior_cache_path = config_to_prior_cache(config, params)
    results_path = join(HQ_CACHE_PATH, run_name,
                        'thejoker-samples.hdf5')
    tasks_path = join(HQ_CACHE_PATH, run_name, 'tmp-tasks.hdf5')

    if not exists(prior_cache_path):
        raise IOError("Prior cache file '{0}' does not exist! Did you run "
                      "make_prior_cache.py?")

    if not exists(tasks_path):
        raise IOError("Tasks file '{0}' does not exist! Did you run "
                      "make_tasks.py?")

    with h5py.File(results_path, 'a') as f:  # ensure the file exists
        done_apogee_ids = list(f.keys())
    if overwrite:
        done_apogee_ids = list()

    # Get data files out of config file:
    allstar, allvisit = config_to_alldata(config)
    allstar = allstar[~np.isin(allstar['APOGEE_ID'], done_apogee_ids)]
    allvisit = allvisit[np.isin(allvisit['APOGEE_ID'], allstar['APOGEE_ID'])]

    # Create TheJoker sampler instance with the specified random seed and pool
    rnd = np.random.RandomState(seed=seed)

    logger.debug("Creating TheJoker instance with {0}".format(rnd))
    joker = TheJoker(params, random_state=rnd, n_batches=8)  # HACK: MAGICNUMBER
    logger.debug("Processing pool has size = {0}".format(pool.size))
    logger.debug("{0} stars left to process for run '{1}'"
                 .format(len(allstar), run_name))

    with h5py.File(results_path, 'a') as results_f:
        processed_ids = list(results_f.keys())

    tasks = []
    with h5py.File(tasks_path, 'r') as tasks_f:
        for apogee_id in tasks_f:
            data = RVData.from_hdf5(tasks_f[apogee_id])
            tasks.append((apogee_id, data))

    logger.debug("Loading data and preparing tasks...")
    full_tasks = []
    for apogee_id, data in tasks:
        if apogee_id in processed_ids and not overwrite:
            continue

        full_tasks.append([joker, apogee_id, data, config, results_path])

    logger.info('Done preparing tasks: {0} stars in process queue'
                .format(len(tasks)))

    for r in tqdm(pool.starmap(worker, full_tasks, callback=callback),
                  total=len(full_tasks)):
        pass


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='Run The Joker on APOGEE data',
                        loggers=[logger, joker_logger])

    parser.add_argument("--name", dest="run_name", required=True,
                        type=str, help="The name of the run.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
                        help="Random number seed")

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true",
                        help="Overwrite any existing samplings")

    args = parser.parse_args()

    if args.mpi:
        Pool = MPIAsyncPool
    else:
        Pool = SerialPool

    with Pool() as pool:
        main(run_name=args.run_name, pool=pool, overwrite=args.overwrite,
             seed=args.seed)

    sys.exit(0)
