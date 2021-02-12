# Standard library
import os
from os.path import join
import sys

# Third-party
import astropy.table as at
import numpy as np
import tables as tb
from tqdm import tqdm
import thejoker as tj

# Project
from hq.log import logger
from hq.config import Config


def worker(task):
    apogee_id, results_path, output_path, config = task

    with tb.open_file(results_path, 'r') as f:
        samples = tj.JokerSamples.read(f.root[apogee_id])

        # more_cols = dict()
        # if ('ln_likelihood' in f[apogee_id].keys() and
        #         f[apogee_id + '/ln_likelihood'].shape != ()):
        #     more_cols['ln_likelihood'] = f[apogee_id]['ln_likelihood'][:]
        #     more_cols['ln_prior'] = f[apogee_id]['ln_prior'][:]
        # else:
        #     more_cols['ln_likelihood'] = np.full(len(samples), np.nan)
        #     more_cols['ln_prior'] = np.full(len(samples), np.nan)

    # for k in more_cols:
    #     res[k] = more_cols[k]

    samples.write(join(output_path, apogee_id[:4], f'{apogee_id}.fits'),
                  overwrite=True)


def main(run_name, pool, overwrite=False):
    c = Config.from_run_name(run_name)
    logger.debug(f'Loaded config for run {run_name}')

    meta = at.Table.read(c.metadata_file)

    samples_path = join(c.run_path, 'samples')
    logger.debug(f'Writing samples to {samples_path}')

    unq_stubs = np.unique([x[:4] for x in meta['APOGEE_ID']])
    for stub in unq_stubs:
        os.makedirs(join(samples_path, stub), exist_ok=True)

    tasks = []

    logger.debug('Preparing APOGEE IDs using metadata file')

    for row in meta:
        apogee_id = str(row['APOGEE_ID'])
        if os.path.exists(join(samples_path, apogee_id[:4],
                               f'{apogee_id}.fits')):
            continue

        if row['mcmc_success']:
            tasks.append((apogee_id, c.mcmc_results_file,
                          samples_path, c))
        else:
            tasks.append((apogee_id, c.joker_results_file,
                          samples_path, c))

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
