# Standard library
import os

# Third-party
import h5py
import numpy as np
import theano
theano.config.optimizer = 'None'
from thejoker.utils import batch_tasks

# Project
from hq.config import Config
from hq.log import logger
from hq.script_helpers import get_parser


def _prior_cache_worker(task):
    (i1, i2), task_id, prior, global_random_state = task
    n_samples = i2 - i1
    if n_samples <= 0:
        return None

    if global_random_state is not None:
        random_state = np.random.RandomState()
        random_state.set_state(global_random_state.get_state())
        random_state.seed(task_id)  # TODO: is this safe?

    samples = prior.sample(size=n_samples, return_logprobs=True)
    return samples


def main(name, pool, overwrite, seed, n_batches=None):
    c = Config.from_run_name(name)
    random_state = np.random.RandomState(seed)

    if os.path.exists(c.prior_cache_file) and not overwrite:
        logger.debug(f"Prior cache file already exists at "
                     f"{c.prior_cache_file}! Use -o / --overwrite "
                     f"to re-generate.")
        return

    logger.debug(f"Prior samples file not found: generating "
                 f"{c.n_prior_samples} samples in cache file at "
                 f"{c.prior_cache_file}")

    if n_batches is None:
        n_batches = max(pool.size, 1)

    tasks = batch_tasks(c.n_prior_samples, n_batches,
                        args=(c.prior, random_state))

    all_samples = []
    for samples in pool.map(_prior_cache_worker, tasks):
        if samples is not None:
            all_samples.append(samples)

    with h5py.File(c.prior_cache_file, 'w') as f:
        for i, samples in enumerate(all_samples):
            samples.write(f, append=i > 0)

    logger.debug("...done generating cache.")


if __name__ == "__main__":
    # Define parser object
    parser = get_parser(description='Generate an HQ run template with the '
                                    'specified name. After running this, you '
                                    'then have to go in to the run directory '
                                    'and edit the configuration.',
                        loggers=logger)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
                        help="Random number seed")

    args = parser.parse_args()

    if args.mpi:
        from schwimmbad.mpi import MPIAsyncPool
        Pool = MPIAsyncPool
        kw = dict()
    elif args.n_procs > 1:
        from schwimmbad import MultiPool
        Pool = MultiPool
        kw = dict(processes=args.n_procs)
    else:
        from schwimmbad import SerialPool
        Pool = SerialPool
        kw = dict()

    args = parser.parse_args()

    with Pool(**kw) as pool:
        main(args.run_name, pool=pool, overwrite=args.overwrite,
             seed=args.seed)
