# Standard library
import atexit
import os
import socket
import sys
import time

# Third-party
# Third-party
import pytensor

pytensor.config.optimizer = "None"
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.reoptimize_unpickled_function = False
pytensor.config.cxx = ""
import h5py
import numpy as np
import thejoker as tj
from astropy.table import QTable
from hq.config import Config

# Project
from hq.log import logger
from hq.samples_analysis import extract_MAP_sample
from run_apogee import callback, tmpdir_combine
from thejoker.logging import logger as joker_logger
from thejoker.multiproc_helpers import batch_tasks
from thejoker.utils import read_batch


def worker(task):
    apogee_ids, worker_id, c, results_path, prior, tmpdir, global_rnd = task

    # This worker's results:
    results_filename = os.path.join(tmpdir, f"worker-{worker_id}.hdf5")
    metadata = QTable.read(c.metadata_file)

    rnd = global_rnd.seed(worker_id)
    logger.log(1, f"Worker {worker_id}: Creating TheJoker instance with {rnd}")
    prior = c.get_prior()
    joker = tj.TheJoker(prior, random_state=rnd)
    logger.debug(
        f"Worker {worker_id} on node {socket.gethostname()}: "
        f"{len(apogee_ids)} stars left to process"
    )

    # Initialize to get packed column order:
    logger.log(
        1,
        f"Worker {worker_id}: Loading prior samples from cache "
        f"{c.prior_cache_file}",
    )
    with h5py.File(c.tasks_file, "r") as tasks_f:
        data = tj.RVData.from_timeseries(tasks_f[apogee_ids[0]])
    joker_helper = joker._make_joker_helper(data)
    _slice = slice(0, c.max_prior_samples, 1)
    batch = read_batch(
        c.prior_cache_file,
        joker_helper.packed_order,
        slice_or_idx=_slice,
        units=joker_helper.internal_units,
    )
    ln_prior = read_batch(c.prior_cache_file, ["ln_prior"], _slice)[:, 0]
    logger.log(1, f"Worker {worker_id}: Loaded {len(batch)} prior samples")

    for apogee_id in apogee_ids:
        if apogee_id not in metadata["APOGEE_ID"]:
            logger.debug(f"{apogee_id} not found in metadata file!")
            continue

        with h5py.File(c.tasks_file, "r") as tasks_f:
            data = tj.RVData.from_timeseries(tasks_f[apogee_id])

        # Subtract out MAP sample, run on residual:
        metadata_row = metadata[metadata["APOGEE_ID"] == apogee_id]
        MAP_sample = extract_MAP_sample(metadata_row)
        orbit = MAP_sample.get_orbit(0)
        new_rv = data.rv - orbit.radial_velocity(data.t)
        data = tj.RVData(t=data.t, rv=new_rv, rv_err=data.rv_err)
        logger.debug(
            f"Worker {worker_id}: Running {apogee_id} " f"({len(data)} visits)"
        )

        t0 = time.time()
        try:
            samples = joker.iterative_rejection_sample(
                data=data,
                n_requested_samples=c.requested_samples_per_star,
                prior_samples=batch,
                init_batch_size=250_000,
                growth_factor=32,
                randomize_prior_order=c.randomize_prior_order,
                return_logprobs=ln_prior,
                in_memory=True,
            )
        except Exception as e:
            logger.warning(f"\t Failed sampling for star {apogee_id} " f"\n Error: {e}")
            continue

        dt = time.time() - t0
        logger.debug(
            f"Worker {worker_id}: {apogee_id} ({len(data)} visits): "
            f"done sampling - {len(samples)} raw samples returned "
            f"({dt:.2f} seconds)"
        )

        # Ensure only positive K values
        samples.wrap_K()

        with h5py.File(results_filename, "a") as results_f:
            if apogee_id in results_f:
                del results_f[apogee_id]
            g = results_f.create_group(apogee_id)
            samples.write(g)

    result = {
        "tmp_filename": results_filename,
        "joker_results_file": results_path,
        "hostname": socket.gethostname(),
        "worker_id": worker_id,
    }
    return result


def main(run_name, pool, overwrite=False, seed=None):
    c = Config.from_run_name(run_name)

    # Get paths to files needed to run
    results_path = os.path.join(c.run_path, "thejoker-control.hdf5")

    # Make directory for temp. files, one per worker:
    tmpdir = os.path.join(c.run_path, "null-control")
    if os.path.exists(tmpdir):
        logger.warning(
            f"Stale temp. file directory found at {tmpdir}: " "combining files first..."
        )
        tmpdir_combine(tmpdir, results_path)

    # ensure the results file exists
    logger.debug("Loading past results...")
    with h5py.File(results_path, "a") as f:
        done_apogee_ids = list(f.keys())
    if overwrite:
        done_apogee_ids = list()

    # Get data files out of config file:
    logger.debug("Loading data...")
    allstar, _ = c.load_alldata()
    allstar = allstar[~np.isin(allstar["APOGEE_ID"], done_apogee_ids)]

    # Create TheJoker sampler instance with the specified random seed and pool
    rnd = np.random.RandomState(seed=seed)
    logger.debug(f"Processing pool has size = {pool.size}")

    apogee_ids = np.unique(allstar["APOGEE_ID"])

    if done_apogee_ids:
        logger.info(
            f"{len(done_apogee_ids)} already completed - "
            f"{len(apogee_ids)} left to process"
        )

    # Load the prior:
    logger.debug("Creating JokerPrior instance...")
    prior = c.get_prior()

    os.makedirs(tmpdir)
    atexit.register(tmpdir_combine, tmpdir, results_path)

    logger.debug("Preparing tasks...")
    if len(apogee_ids) > 10 * pool.size:
        n_tasks = min(16 * pool.size, len(apogee_ids))
    else:
        n_tasks = pool.size
    tasks = batch_tasks(
        len(apogee_ids),
        n_tasks,
        arr=apogee_ids,
        args=(c, results_path, prior, tmpdir, rnd),
    )

    logger.info(f"Done preparing tasks: split into {len(tasks)} task chunks")
    for r in pool.map(worker, tasks, callback=callback):
        pass


if __name__ == "__main__":
    from hq.script_helpers import get_parser
    from threadpoolctl import threadpool_limits

    # Define parser object
    parser = get_parser(
        description="Run The Joker on APOGEE data", loggers=[logger, joker_logger]
    )

    parser.add_argument(
        "-s", "--seed", dest="seed", default=None, type=int, help="Random number seed"
    )

    args = parser.parse_args()

    seed = args.seed
    if seed is None:
        seed = np.random.randint(2**32 - 1)
        logger.log(1, f"No random number seed specified, so using seed: {seed}")

    with threadpool_limits(limits=1, user_api="blas"):
        with args.Pool(**args.Pool_kwargs) as pool:
            main(
                run_name=args.run_name,
                pool=pool,
                overwrite=args.overwrite,
                seed=args.seed,
            )

    sys.exit(0)
