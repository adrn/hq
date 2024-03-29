import os

import h5py
import numpy as np
import pytensor

pytensor.config.optimizer = "None"
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.reoptimize_unpickled_function = False
pytensor.config.cxx = ""
from thejoker.utils import batch_tasks

from hq.config import Config
from hq.log import logger


def _prior_cache_worker(task):
    (i1, i2), task_id, prior, global_random_state = task
    n_samples = i2 - i1
    if n_samples <= 0:
        return None

    logger.debug(f"Worker {task_id} generating {n_samples} samples")

    if global_random_state is not None:
        seed = global_random_state.integers(0, 2**28) + task_id
        random_state = np.random.default_rng(seed)
    else:
        random_state = None

    samples = prior.sample(
        size=n_samples,
        return_logprobs=True,
        dtype=np.float32,
        random_state=random_state,
    )

    logger.debug(f"Worker {task_id} finished")

    return samples


def make_prior_cache(run_path, pool, overwrite, seed, n_batches=None):
    c = Config(run_path / "config.yml")
    c.cache_path.mkdir(exist_ok=True)

    random_state = np.random.default_rng(seed)

    if os.path.exists(c.prior_cache_file) and not overwrite:
        logger.debug(
            f"Prior cache file already exists at "
            f"{c.prior_cache_file}! Use -o / --overwrite "
            f"to re-generate."
        )
        return

    logger.debug(
        f"Prior samples file not found: generating "
        f"{c.n_prior_samples} samples in cache file at "
        f"{c.prior_cache_file}"
    )

    if n_batches is None:
        n_batches = max(pool.size, 1)

    prior, model = c.get_prior()
    prior.sample(size=1)  # initialize
    tasks = batch_tasks(c.n_prior_samples, n_batches, args=(prior, random_state))

    all_samples = []
    for samples in pool.map(_prior_cache_worker, tasks):
        if samples is not None:
            all_samples.append(samples)

    with h5py.File(c.prior_cache_file, "w") as f:
        for i, samples in enumerate(all_samples):
            samples.write(f, append=i > 0)

    logger.debug("...done generating cache.")
