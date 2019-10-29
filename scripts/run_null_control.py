# Standard library
from os.path import join, exists
import sys

# Third-party
import h5py
import numpy as np
from thejoker.log import log as joker_logger
from thejoker.sampler import TheJoker
from thejoker.data import RVData
from tqdm import tqdm
import yaml
from schwimmbad import SerialPool
from schwimmbad.mpi import MPIAsyncPool

# Project
from hq.data import get_rvdata
from hq.log import logger
from hq.config import (HQ_CACHE_PATH, config_to_jokerparams,
                       config_to_prior_cache, config_to_alldata)
from hq.script_helpers import get_parser

from run_apogee import worker, callback


def main(run_name, pool, overwrite=False, seed=None):
    run_path = join(HQ_CACHE_PATH, run_name)
    with open(join(run_path, 'config.yml'), 'r') as f:
        config = yaml.load(f.read())

    # Get paths to files needed to run
    params = config_to_jokerparams(config)
    prior_cache_path = config_to_prior_cache(config, params)
    results_path = join(HQ_CACHE_PATH, run_name,
                        'thejoker-control.hdf5')

    if not exists(prior_cache_path):
        raise IOError("Prior cache file '{0}' does not exist! Did you run "
                      "make_prior_cache.py?")

    with h5py.File(results_path, 'a') as f: # ensure the file exists
        pass

    # Get data files out of config file:
    allstar, allvisit = config_to_alldata(config)

    # HACK: MAGIC NUMBER
    # Reduce the sample size:
    n_control = len(allstar) // 10
    idx = np.random.choice(len(allstar), size=n_control, replace=False)
    allstar = allstar[idx]
    allvisit = allvisit[np.isin(allvisit['APOGEE_ID'], allstar['APOGEE_ID'])]

    # Create TheJoker sampler instance with the specified random seed and pool
    rnd = np.random.RandomState(seed=seed)

    logger.debug("Creating TheJoker instance with {0}".format(rnd))
    joker = TheJoker(params, random_state=rnd, n_batches=8) # HACK: MAGIC NUMBER
    logger.debug("Processing pool has size = {0}".format(pool.size))
    logger.debug("{0} stars left to process for run '{1}'"
                 .format(len(allstar), run_name))

    tasks = []
    logger.debug("Loading data and preparing tasks...")
    for star in tqdm(allstar):
        visits = allvisit[allvisit['APOGEE_ID'] == star['APOGEE_ID']]
        data = get_rvdata(visits)

        # Overwrite the data with a null signal!
        new_rv = rnd.normal(np.mean(data.rv).value,
                            data.stddev.value) * data.rv.unit
        new_data = RVData(rv=new_rv, t=data.t, stddev=data.stddev)

        tasks.append([joker, star['APOGEE_ID'], new_data, config, results_path])

    logger.info('Done preparing tasks: {0} stars in process queue'
                .format(len(tasks)))

    for r in tqdm(pool.starmap(worker, tasks, callback=callback),
                  total=len(tasks)):
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

    parser.add_argument("-s", "--seed", dest="seed", default=42, type=int,
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
