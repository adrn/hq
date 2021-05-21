# Standard library
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
    conf = task['conf']

    conf['output_path'].mkdir(exist_ok=True)

    for row in task['metadata']:
        filename = task['output_path'] / f"{row[conf.source_id_colname]}.fits"
        if filename.exists() and not task['overwrite']:
            continue

        # TODO: should this be configurable?
        if 0 < row['mcmc_status'] <= 2:
            results_file = conf.mcmc_results_file
        else:
            results_file = conf.joker_results_file

        with tb.open_file(results_file, 'r') as f:
            samples = tj.JokerSamples.read(f.root[conf.source_id_colname])

        samples.write(filename, overwrite=True)


def expand_samples(run_path, pool, overwrite=False):
    conf = Config(run_path / 'config.yml')
    logger.debug(f'Loaded config for run {run_path}')

    meta = at.Table.read(conf.metadata_file)

    root_output_path = conf.cache_path / 'samples'
    root_output_path.mkdir(exist_ok=True)
    logger.debug(f'Writing samples to {root_output_path!s}')

    if conf.expand_subdir_column is None:
        raise ValueError("Config item expand_subdir_column must be specified!")

    unq_subdirs = np.unique(meta[conf.expand_subdir_column])
    tasks = []
    for subdir in unq_subdirs:
        sub_meta = meta[meta[conf.expand_subdir_column] == subdir]
        output_path = root_output_path / subdir
        tasks.append({
            'conf': conf,
            'output_path': output_path,
            'metadata': sub_meta,
            'overwrite': overwrite
        })

    for r in tqdm(pool.map(worker, tasks), total=len(tasks)):
        pass
