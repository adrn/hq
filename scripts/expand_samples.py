# Standard library
import os
from os.path import join
import sys

# Third-party
import numpy as np
import h5py
from tqdm import tqdm
import thejoker as tj

# Project
from hq.log import logger
from hq.config import Config


def worker(task):
    apogee_id, results_path, output_path, config = task

    with h5py.File(results_path, 'r') as f:
        samples = tj.JokerSamples.read(f[apogee_id])

        # more_cols = dict()
        # if ('ln_likelihood' in f[apogee_id].keys() and
        #         f[apogee_id + '/ln_likelihood'].shape != ()):
        #     more_cols['ln_likelihood'] = f[apogee_id]['ln_likelihood'][:]
        #     more_cols['ln_prior'] = f[apogee_id]['ln_prior'][:]
        # else:
        #     more_cols['ln_likelihood'] = np.full(len(samples), np.nan)
        #     more_cols['ln_prior'] = np.full(len(samples), np.nan)

    res = samples.to_table()
    # for k in more_cols:
    #     res[k] = more_cols[k]

    res.write(join(output_path, apogee_id[:4], f'{apogee_id}.fits.gz'),
              overwrite=True)


def main(run_name, pool, overwrite=False):
    c = Config.from_run_name(run_name)
    allstar, allvisit = c.load_alldata()

    samples_path = join(c.run_path, 'samples')
    logger.debug(f'Writing samples to {samples_path}')

    unq_stubs = np.unique([x[:4] for x in allstar['APOGEE_ID']])
    for stub in unq_stubs:
        os.makedirs(join(samples_path, stub), exist_ok=True)

    tasks = []

    joker_f = h5py.File(c.joker_results_path, 'r')
    mcmc_f = h5py.File(c.mcmc_results_path, 'r')

    logger.debug('Reading {} APOGEE IDs from Joker samplings'
                 .format(len(joker_f.keys())))

    for apogee_id in joker_f.keys():
        if apogee_id in mcmc_f.keys():
            tasks.append((apogee_id, c.mcmc_results_path,
                          samples_path, c))
        else:
            tasks.append((apogee_id, c.joker_results_path,
                          samples_path, c))

    joker_f.close()
    mcmc_f.close()

    for r in tqdm(pool.map(worker, tasks), total=len(tasks)):
        pass


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    from hq.script_helpers import get_parser

    # Define parser object
    parser = get_parser(description='TODO', loggers=[logger])

    args = parser.parse_args()

    with threadpool_limits(limits=1, user_api='blas'):
        with args.Pool(**args.Pool_kwargs) as pool:
            main(run_name=args.run_name, pool=pool, overwrite=args.overwrite)

    sys.exit(0)
