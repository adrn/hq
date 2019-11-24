# Standard library
import os
from os.path import join
import sys

# Third-party
import numpy as np
import h5py
from tqdm import tqdm
import yaml
from thejoker import JokerSamples
from schwimmbad import SerialPool
from schwimmbad.mpi import MPIAsyncPool

# Project
from hq.log import logger
from hq.config import HQ_CACHE_PATH, config_to_alldata
from hq.script_helpers import get_parser


def worker(apogee_id, results_path, output_path, config):
    with h5py.File(results_path, 'r') as f:
        samples = JokerSamples.from_hdf5(
            f[apogee_id], poly_trend=config['hyperparams']['poly_trend'])

        more_cols = dict()
        if ('ln_likelihood' in f[apogee_id].keys() and
                f[apogee_id + '/ln_likelihood'].shape != ()):
            more_cols['ln_likelihood'] = f[apogee_id]['ln_likelihood'][:]
            more_cols['ln_prior'] = f[apogee_id]['ln_prior'][:]
        else:
            more_cols['ln_likelihood'] = np.full(len(samples), np.nan)
            more_cols['ln_prior'] = np.full(len(samples), np.nan)

    res = samples.to_table()
    for k in more_cols:
        res[k] = more_cols[k]

    if 'jitter' in config['hyperparams']:
        res.remove_column('jitter')

    res.write(join(output_path, apogee_id[:4], '{}.fits.gz'.format(apogee_id)),
              overwrite=True)


def main(run_name, pool, overwrite=False):
    run_path = join(HQ_CACHE_PATH, run_name)
    with open(join(run_path, 'config.yml'), 'r') as f:
        config = yaml.load(f.read())

    allstar, allvisit = config_to_alldata(config)

    samples_path = join(HQ_CACHE_PATH, run_name, 'samples')
    logger.debug(f'Writing samples to {samples_path}')

    unq_stubs = np.unique([x[:4] for x in allstar['APOGEE_ID']])
    for stub in unq_stubs:
        os.makedirs(join(samples_path, stub), exist_ok=True)

    joker_results_path = join(HQ_CACHE_PATH, run_name,
                              'thejoker-samples.hdf5')
    emcee_results_path = join(HQ_CACHE_PATH, run_name,
                              'emcee-samples.hdf5')

    tasks = []

    joker_f = h5py.File(joker_results_path, 'r')
    emcee_f = h5py.File(emcee_results_path, 'r')

    logger.debug('Reading {} APOGEE IDs from Joker samplings'
                 .format(len(joker_f.keys())))

    for apogee_id in joker_f.keys():
        if apogee_id in emcee_f.keys():
            tasks.append((apogee_id, emcee_results_path,
                          samples_path, config))
        else:
            tasks.append((apogee_id, joker_results_path,
                          samples_path, config))

    joker_f.close()
    emcee_f.close()

    for r in tqdm(pool.starmap(worker, tasks), total=len(tasks)):
        pass


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='TODO',
                        loggers=logger)

    parser.add_argument("--name", dest="run_name", required=True,
                        type=str, help="The name of the run.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true",
                        help="Overwrite any existing samplings")

    args = parser.parse_args()

    if args.mpi:
        Pool = MPIAsyncPool
    else:
        Pool = SerialPool

    with Pool() as pool:
        main(run_name=args.run_name, pool=pool)

    sys.exit(0)
