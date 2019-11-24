# Standard library
import atexit
import glob
import os
import shutil
import sys
import time

# Third-party
import theano
theano.config.optimizer = 'None'
theano.config.mode = 'FAST_COMPILE'
theano.config.reoptimize_unpickled_function = False
import h5py
import numpy as np
from thejoker.logging import logger as joker_logger
from thejoker import TheJoker
from thejoker.data import RVData
from thejoker.multiproc_helpers import batch_tasks

# Project
from hq.log import logger
from hq.config import Config
from hq.script_helpers import get_parser


def worker(task):
    apogee_ids, worker_id, c, prior, tmpdir, global_rnd = task

    # This worker's results:
    results_filename = os.path.join(tmpdir, f'worker-{worker_id}.hdf5')

    rnd = global_rnd.seed(worker_id)
    logger.log(1, f"Creating TheJoker instance with {rnd}")
    joker = TheJoker(prior, random_state=rnd)
    logger.debug(f"Worker batch id {worker_id}: "
                 f"{len(apogee_ids)} stars left to process")

    for apogee_id in apogee_ids:
        with h5py.File(c.tasks_path, 'r') as tasks_f:
            data = RVData.from_timeseries(tasks_f[apogee_id])

        t0 = time.time()
        logger.log(0, "{0}: Starting sampling".format(apogee_id))

        try:
            samples = joker.iterative_rejection_sample(
                data=data, n_requested_samples=c.requested_samples_per_star,
                prior_samples=c.prior_cache_file,
                randomize_prior_order=c.randomize_prior_order,
                return_logprobs=True)
        except Exception as e:
            logger.warning(f"\t Failed sampling for star {apogee_ids} "
                           f"\n Error: {e}")
            continue

        dt = time.time() - t0
        logger.log(1,
                   f"{apogee_id}: done sampling - {len(samples)} raw samples "
                   f"returned ({dt:.2f} seconds)")

        # Ensure only positive K values
        samples.wrap_K()

        with h5py.File(results_filename, 'a') as results_f:
            if apogee_id in results_f:
                del results_f[apogee_id]
            g = results_f.create_group(apogee_id)
            samples.write(g)


def tmpdir_combine(tmpdir, results_filename):
    logger.debug(f"Combining results into {results_filename}")
    tmp_files = sorted(glob.glob(os.path.join(tmpdir, '*.hdf5')))
    with h5py.File(results_filename, 'a') as all_f:
        for tmp_file in tmp_files:
            with h5py.File(tmp_file, 'r') as f:
                for key in f:
                    if key in all_f:
                        del all_f[key]
                    f.copy(key, all_f)
            os.remove(tmp_file)
    shutil.rmtree(tmpdir)


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
        logger.debug(f"{len(done_apogee_ids)} already completed")

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

    # Make directory for temp. files, one per worker:
    tmpdir = os.path.join(c.run_path, 'thejoker')
    if os.path.exists(tmpdir):
        logger.warning(f"Stale temp. file directory found at {tmpdir}: "
                       "removing...")
        time.sleep(3)
        shutil.rmtree(tmpdir)
    os.makedirs(tmpdir)
    atexit.register(tmpdir_combine, tmpdir, c.joker_results_path)

    logger.debug("Preparing tasks...")
    tasks = batch_tasks(len(apogee_ids), pool.size, arr=apogee_ids,
                        args=(c, prior, tmpdir, rnd))

    logger.info(f'Done preparing tasks: split into {len(tasks)} task chunks')
    for r in pool.map(worker, tasks):
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
        logger.debug(f"No random number seed specified, so using seed: {seed}")

    with args.Pool(**args.Pool_kwargs) as pool:
        main(run_name=args.run_name, pool=pool, overwrite=args.overwrite,
             seed=args.seed, limit=args.limit)

    sys.exit(0)
