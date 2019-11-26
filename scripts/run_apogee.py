# Standard library
import os
import sys
import time

# Third-party
import theano
theano.config.optimizer = 'None'
theano.config.mode = 'FAST_COMPILE'
theano.config.reoptimize_unpickled_function = False
theano.config.cxx = ""
import h5py
import numpy as np
from thejoker.logging import logger as joker_logger
import thejoker as tj

# Project
from hq.log import logger
from hq.config import Config
from hq.script_helpers import get_parser


def worker(task):
    apogee_id, worker_id, c, prior, global_rnd = task

    rnd = global_rnd.seed(worker_id)
    logger.log(1, f"Creating TheJoker instance with {rnd}")
    joker = tj.TheJoker(prior, random_state=rnd)

    logger.log(1, f"Running {apogee_id}")
    with h5py.File(c.tasks_path, 'r') as tasks_f:
        data = tj.RVData.from_timeseries(tasks_f[apogee_id])

    t0 = time.time()
    logger.log(1, f"{apogee_id}: Starting sampling")

    try:
        samples = joker.iterative_rejection_sample(
            data=data, n_requested_samples=c.requested_samples_per_star,
            prior_samples=c.prior_cache_file,
            randomize_prior_order=c.randomize_prior_order,
            return_logprobs=True)
    except Exception as e:
        logger.warning(f"\t Failed sampling for star {apogee_id} "
                       f"\n Error: {e}")
        return None

    dt = time.time() - t0
    logger.debug(f"{apogee_id} ({len(data)} visits): done sampling - "
                 f"{len(samples)} raw samples returned ({dt:.2f} seconds)")

    # Ensure only positive K values
    samples.wrap_K()

    result = {'apogee_id': apogee_id,
              'samples': samples,
              'joker_results_path': c.joker_results_path}
    return result


def callback(result):
    if result is None:
        return

    samples = result['samples']
    joker_results_file = result['joker_results_path']
    apogee_id = result['apogee_id']

    with h5py.File(joker_results_file, 'a') as f:
        if apogee_id in f:
            del f[apogee_id]
        g = f.create_group(apogee_id)
        samples.write(g)


def main(run_name, pool, overwrite=False, seed=None, limit=None):
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

    if done_apogee_ids:
        logger.info(f"{len(done_apogee_ids)} already completed")

    # Get data files out of config file:
    logger.debug("Loading data...")
    allstar, _ = c.load_alldata()
    allstar = allstar[~np.isin(allstar['APOGEE_ID'], done_apogee_ids)]

    # Create TheJoker sampler instance with the specified random seed and pool
    rnd = np.random.RandomState(seed=seed)
    logger.debug(f"Processing pool has size = {pool.size}")

    apogee_ids = np.unique(allstar['APOGEE_ID'])
    if limit is not None:
        apogee_ids = apogee_ids[:limit]

    # Load the prior:
    logger.debug("Creating JokerPrior instance...")
    prior = c.get_prior()

    logger.debug("Preparing tasks...")
    tasks = [(apogee_id, i, c, prior, rnd)
             for i, apogee_id in enumerate(apogee_ids)]
    logger.info(f'Done preparing tasks: split into {len(tasks)} task chunks')
    for r in pool.map(worker, tasks, callback=callback):
        pass


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='Run The Joker on APOGEE data',
                        loggers=[logger, joker_logger])

    parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
                        help="Random number seed")

    parser.add_argument("--limit", dest="limit", default=None,
                        type=int, help="Maximum number of stars to process")

    args = parser.parse_args()

    seed = args.seed
    if seed is None:
        seed = np.random.randint(2**32 - 1)
        logger.log(1, f"No random number seed specified, so using seed: {seed}")

    with args.Pool(**args.Pool_kwargs) as pool:
        main(run_name=args.run_name, pool=pool, overwrite=args.overwrite,
             seed=args.seed, limit=args.limit)

    sys.exit(0)
