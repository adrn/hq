# Standard library
import os
import sys
import time

# Third-party
import h5py
import numpy as np
import theano
theano.config.optimizer = 'None'
from thejoker.logging import logger as joker_logger
from thejoker import TheJoker
from tqdm import tqdm
from thejoker.data import RVData

# Project
from hq.log import logger
from hq.config import Config
from hq.script_helpers import get_parser


def worker(task):
    joker, apogee_id, data, c = task

    t0 = time.time()
    logger.log(1, "{0}: Starting sampling".format(apogee_id))

    try:
        samples = joker.iterative_rejection_sample(
            data=data, n_requested_samples=c.requested_samples_per_star,
            prior_samples=c.prior_cache_file,
            randomize_prior_order=c.randomize_prior_order,
            return_logprobs=True)
    except Exception as e:
        logger.warning("\t Failed sampling for star {0} \n Error: {1}"
                       .format(apogee_id, str(e)))
        return None

    logger.debug("{0}: done sampling - {1} raw samples returned "
                 "({2:.2f} seconds)".format(apogee_id, len(samples),
                                            time.time() - t0))

    # Ensure only positive K values
    samples.wrap_K()

    res = dict()
    res['apogee_id'] = apogee_id
    res['samples'] = samples
    res['results_filename'] = c.joker_results_path
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
        res['samples'].write(g)

    logger.debug("{0}: done, {1} samples returned ".format(res['apogee_id'],
                                                           len(res['samples'])))


def main(run_name, pool, overwrite=False, seed=None):
    c = Config.from_run_name(run_name)

    if not os.path.exists(c.prior_cache_file):
        raise IOError(f"Prior cache file {c.prior_cache_file} does not exist! "
                      "Did you run make_prior_cache.py?")

    if not os.path.exists(c.tasks_path):
        raise IOError("Tasks file '{0}' does not exist! Did you run "
                      "make_tasks.py?")

    # ensure the results file exists
    with h5py.File(c.joker_results_path, 'a') as f:
        done_apogee_ids = list(f.keys())
    if overwrite:
        done_apogee_ids = list()

    # Get data files out of config file:
    allstar, allvisit = c.load_alldata()
    allstar = allstar[~np.isin(allstar['APOGEE_ID'], done_apogee_ids)]
    allvisit = allvisit[np.isin(allvisit['APOGEE_ID'], allstar['APOGEE_ID'])]

    # Create TheJoker sampler instance with the specified random seed and pool
    rnd = np.random.RandomState(seed=seed)

    logger.debug(f"Creating TheJoker instance with {rnd}")
    joker = TheJoker(c.prior, random_state=rnd)
    logger.debug(f"Processing pool has size = {pool.size}")
    logger.debug(f"{len(allstar)} stars left to process for run {c.name}")

    with h5py.File(c.joker_results_path, 'a') as results_f:
        processed_ids = list(results_f.keys())

    tasks = []
    with h5py.File(c.tasks_path, 'r') as tasks_f:
        for apogee_id in tasks_f:
            data = RVData.from_timeseries(tasks_f[apogee_id])
            tasks.append((apogee_id, data))

    logger.debug("Loading data and preparing tasks...")
    full_tasks = []
    for apogee_id, data in tasks:
        if apogee_id in processed_ids and not overwrite:
            continue

        full_tasks.append([joker, apogee_id, data, c])

    logger.info('Done preparing tasks: {0} stars in process queue'
                .format(len(tasks)))

    for r in tqdm(pool.map(worker, full_tasks, callback=callback),
                  total=len(full_tasks)):
        pass


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='Run The Joker on APOGEE data',
                        loggers=[logger, joker_logger])

    parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
                        help="Random number seed")

    args = parser.parse_args()

    seed = args.seed
    if seed is None:
        seed = np.random.randint(2**32 - 1)
        logger.debug(f"No random number seed specified, so using seed: {seed}")

    with args.Pool(**args.Pool_kwargs) as pool:
        main(run_name=args.run_name, pool=pool, overwrite=args.overwrite,
             seed=args.seed)

    sys.exit(0)
