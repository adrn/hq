# Standard library
from os.path import join, exists
import sys

# Third-party
import astropy.table as at
import astropy.units as u
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
from hq.samples_analysis import extract_MAP_orbit

from run_apogee import worker, callback


def get_orbit_samples(n_samples, joker):
    prior_samples = joker.sample_prior(n_samples)
    # TODO HACK MAGIC NUMBER
    K0 = 25.  # km/s
    P = prior_samples['P'].to_value(u.day)
    varK = K0**2 / (1 - prior_samples['e']**2) * (P / 365.)**(-2/3.)
    K = np.random.normal(0., np.sqrt(varK), size=n_samples)
    prior_samples['K'] = K * u.km/u.s
    prior_samples['v0'] = np.zeros(len(K)) * u.km/u.s
    return prior_samples


def main(run_name, pool, overwrite=False, seed=None):
    run_path = join(HQ_CACHE_PATH, run_name)
    with open(join(run_path, 'config.yml'), 'r') as f:
        config = yaml.load(f.read())

    # Get paths to files needed to run
    params = config_to_jokerparams(config)
    prior_cache_path = config_to_prior_cache(config, params)
    results_path = join(HQ_CACHE_PATH, run_name, 'thejoker-injected.hdf5')

    if not exists(prior_cache_path):
        raise IOError("Prior cache file '{0}' does not exist! Did you run "
                      "make_prior_cache.py?")

    with h5py.File(results_path, 'a') as f:  # ensure the file exists
        pass

    # Get data files out of config file:
    allstar, allvisit = config_to_alldata(config)

    # Read metadata file:
    meta_file = join(HQ_CACHE_PATH, run_name, 'metadata-master.fits')
    meta = at.Table.read(meta_file)
    master = at.join(meta, allstar, keys='APOGEE_ID')

    # n_control = len(allstar) // 10
    n_control = 4  # TODO: remove this when running in production
    idx = np.random.choice(len(master), size=n_control, replace=False)
    master = master[idx]
    allvisit = allvisit[np.isin(allvisit['APOGEE_ID'], master['APOGEE_ID'])]

    # Create TheJoker sampler instance with the specified random seed and pool
    rnd = np.random.RandomState(seed=seed)

    logger.debug("Creating TheJoker instance with {0}".format(rnd))
    joker = TheJoker(params, random_state=rnd)
    logger.debug("Processing pool has size = {0}".format(pool.size))
    logger.debug("{0} stars left to process for run '{1}'"
                 .format(len(master), run_name))

    # generate companion orbital parameters
    companion_samples = get_orbit_samples(n_control, joker)

    tasks = []
    logger.debug("Loading data and preparing tasks...")
    for i, star in enumerate(tqdm(master)):
        visits = allvisit[allvisit['APOGEE_ID'] == star['APOGEE_ID']]
        data = get_rvdata(visits)

        # Get MAP orbit from metadata file:
        orbit = extract_MAP_orbit(star)
        flat_rv = data.rv - orbit.radial_velocity(data.t)
        base_rv = np.median(data.rv) + flat_rv

        # Inject a companion signal:
        orbit = companion_samples.get_orbit(i)
        new_rv = base_rv + orbit.radial_velocity(data.t)

        # Observe the companion signal:
        new_rv = rnd.normal(new_rv.value,
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
