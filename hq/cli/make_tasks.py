# Third-party
import h5py
import numpy as np
from tqdm import tqdm

# Project
from hq.log import logger
from hq.config import Config


def make_tasks(run_path, pool, overwrite=False):
    c = Config(run_path / 'config.yml')

    if c.tasks_file.exists() and not overwrite:
        logger.info(f"File {str(c.tasks_file)} already exists: Use "
                    "-o/--overwrite if needed")
        return

    # Load the full data file, but store only some columns for speed:
    keys = [c.source_id_colname,
            c.rv_colname,
            c.rv_error_colname,
            c.time_colname]
    c._cache['data'] = c.data[keys]

    logger.debug("Loading data and preparing tasks...")
    source_ids = np.unique(c.data[c.source_id_colname])
    with h5py.File(c.tasks_file, 'w') as f:
        for source_id in tqdm(source_ids):
            rvdata = c.get_source_data(source_id)
            g = f.create_group(source_id)
            rvdata.to_timeseries().write(g, format='hdf5', serialize_meta=True)

    logger.info(f'Done preparing tasks: {len(source_ids)} sources in the '
                'processing queue')
