# Standard library
import atexit
import glob
import os
import shutil
import socket
import time

# Third-party
from astropy.utils import iers
iers.conf.auto_download = False

import theano
theano.config.optimizer = 'None'
theano.config.mode = 'FAST_COMPILE'
theano.config.reoptimize_unpickled_function = False
theano.config.cxx = ""
import h5py
import numpy as np
import thejoker as tj
from thejoker.multiproc_helpers import batch_tasks
from thejoker.utils import read_batch

# Project
from hq.log import logger
from hq.config import Config


def worker(task):
    source_ids, worker_id, c, prior, tmpdir, global_rnd = task

    # This worker's results:
    results_filename = os.path.join(tmpdir, f'worker-{worker_id}.hdf5')

    rnd = global_rnd.seed(worker_id)
    logger.log(1, f"Worker {worker_id}: Creating TheJoker instance with {rnd}")
    prior = c.get_prior()
    joker = tj.TheJoker(prior, random_state=rnd)
    logger.debug(f"Worker {worker_id} on node {socket.gethostname()}: "
                 f"{len(source_ids)} stars left to process")

    # Initialize to get packed column order#
    logger.log(1,
               f"Worker {worker_id}: Loading prior samples from cache "
               f"{c.prior_cache_file}")

    # Also pre-load all data to avoid firing off so many file I/O operations:
    all_data = {}
    with h5py.File(c.tasks_path, 'r') as tasks_f:
        for source_id in source_ids:
            data = tj.RVData.from_timeseries(tasks_f[source_id])
            all_data[source_id] = data
    joker_helper = joker._make_joker_helper(data)
    _slice = slice(0, c.max_prior_samples, 1)
    batch = read_batch(c.prior_cache_file, joker_helper.packed_order,
                       slice_or_idx=_slice,
                       units=joker_helper.internal_units)
    ln_prior = read_batch(c.prior_cache_file, ['ln_prior'], _slice)[:, 0]
    logger.log(1, f"Worker {worker_id}: Loaded {len(batch)} prior samples")

    for source_id in source_ids:
        data = all_data[source_id]
        logger.debug(f"Worker {worker_id}: Running {source_id} "
                     f"({len(data)} visits)")

        t0 = time.time()
        try:
            samples = joker.iterative_rejection_sample(
                data=data, n_requested_samples=c.requested_samples_per_star,
                prior_samples=batch,
                init_batch_size=c.init_batch_size,
                growth_factor=32,
                randomize_prior_order=c.randomize_prior_order,
                return_logprobs=ln_prior, in_memory=True)
        except Exception as e:
            logger.warning(f"\t Failed sampling for star {source_id} "
                           f"\n Error: {e}")
            continue

        dt = time.time() - t0
        logger.debug(f"Worker {worker_id}: {source_id} ({len(data)} visits): "
                     f"done sampling - {len(samples)} raw samples returned "
                     f"({dt:.2f} seconds)")

        # Ensure only positive K values
        samples.wrap_K()

        with h5py.File(results_filename, 'a') as results_f:
            if source_id in results_f:
                del results_f[source_id]
            g = results_f.create_group(source_id)
            samples.write(g)

    result = {'tmp_filename': results_filename,
              'joker_results_path': c.joker_results_path,
              'hostname': socket.gethostname(),
              'worker_id': worker_id}
    return result


def callback(result):
    tmp_file = result['tmp_filename']
    joker_results_file = result['joker_results_path']

    logger.debug(f"Worker {result['worker_id']} on {result['hostname']}: "
                 f"Combining results from {tmp_file} into {joker_results_file}"
                 )

    with h5py.File(joker_results_file, 'a') as all_f:
        with h5py.File(tmp_file, 'r') as f:
            for key in f:
                if key in all_f:
                    del all_f[key]
                f.copy(key, all_f)
    os.remove(tmp_file)


def tmpdir_combine(tmpdir, results_filename):
    logger.debug(f"Combining results into {results_filename}")
    tmp_files = sorted(glob.glob(os.path.join(tmpdir, '*.hdf5')))
    with h5py.File(results_filename, 'a') as all_f:
        for tmp_file in tmp_files:
            logger.log(1, f"Processing {tmp_file}")
            with h5py.File(tmp_file, 'r') as f:
                for key in f:
                    if key in all_f:
                        del all_f[key]
                    f.copy(key, all_f)
            os.remove(tmp_file)
    shutil.rmtree(tmpdir)


def run_thejoker(run_path, pool, overwrite=False, seed=None, limit=None):
    c = Config(run_path / 'config.yml')

    if not c.prior_cache_file.exists():
        raise IOError(f"Prior cache file {str(c.prior_cache_file)} does not "
                      "exist! Did you run hq make_prior_cache?")

    if not c.tasks_path.exists():
        raise IOError(f"Tasks file '{str(c.tasks_path)}' does not exist! Did "
                      "you run hq make_tasks?")

    # Make directory for temp. files, one per worker:
    tmpdir = c.cache_path / 'thejoker-tmp'
    print('yo fuck', tmpdir)
    if tmpdir.exists():
        logger.warning(f"Stale temp. file directory found at {str(tmpdir)}: "
                       "combining files first...")
        tmpdir_combine(tmpdir, c.joker_results_path)

    # ensure the results file exists
    logger.debug("Loading past results...")
    with h5py.File(c.joker_results_path, 'a') as f:
        done_source_ids = list(f.keys())
    if overwrite:
        done_source_ids = list()

    # Get data files out of config file:
    logger.debug("Loading data...")
    data = c.data[~np.isin(c.data[c.source_id_colname], done_source_ids)]

    # Create TheJoker sampler instance with the specified random seed and pool
    rnd = np.random.RandomState(seed=seed)
    logger.debug(f"Processing pool has size = {pool.size}")

    source_ids = np.unique(data[c.source_id_colname])
    if limit is not None:
        source_ids = source_ids[:limit]

    if done_source_ids:
        logger.info(f"{len(done_source_ids)} already completed - "
                    f"{len(source_ids)} left to process")

    # Load the prior:
    logger.debug("Creating JokerPrior instance...")
    prior = c.get_prior()

    tmpdir.mkdir(exist_ok=True)
    atexit.register(tmpdir_combine, tmpdir, c.joker_results_path)

    logger.debug("Preparing tasks...")
    if len(source_ids) > 10 * pool.size:
        n_tasks = min(16 * pool.size, len(source_ids))
    else:
        n_tasks = pool.size
    tasks = batch_tasks(len(source_ids), n_tasks, arr=source_ids,
                        args=(c, prior, tmpdir, rnd))

    logger.info(f'Done preparing tasks: split into {len(tasks)} task chunks')
    for r in pool.map(worker, tasks, callback=callback):
        pass
