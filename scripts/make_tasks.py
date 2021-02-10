# Standard library
from os import path
import sys

# Third-party
import h5py
import numpy as np
from thejoker.logging import logger as joker_logger
from tqdm import tqdm
from schwimmbad import SerialPool

# Project
from hq.data import get_rvdata
from hq.log import logger
from hq.config import Config
from hq.script_helpers import get_parser


def main(run_name, pool, overwrite=False):
    c = Config.from_run_name(run_name)

    if path.exists(c.tasks_path) and not overwrite:
        logger.info(f"File {c.tasks_path} already exists: Use -o/--overwrite "
                    "if needed")
        return

    # Load the full allvisit file, but store only some columns for speed:
    allvisit = c.allvisit
    keys = ['APOGEE_ID', 'JD', 'VHELIO',  # 'VISITID'
            'PLATE', 'MJD', 'FIBERID']  # TODO: HACK FOR DR17 Alpha
    c._cache['allvisit'] = allvisit[keys]

    logger.debug("Loading data and preparing tasks...")
    apogee_ids = np.unique(c.allstar['APOGEE_ID'])
    with h5py.File(c.tasks_path, 'w') as f:
        for apogee_id in tqdm(apogee_ids):
            data = c.get_star_data(apogee_id)
            visits = allvisit[allvisit['APOGEE_ID'] == apogee_id]
            data = get_rvdata(visits)

            g = f.create_group(apogee_id)
            data.to_timeseries().write(g, format='hdf5', serialize_meta=True)

    logger.info(f'Done preparing tasks: {len(apogee_ids)} stars in the '
                'processing queue')


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='Run The Joker on APOGEE data',
                        loggers=[logger, joker_logger])

    args = parser.parse_args()

    Pool = SerialPool

    with Pool() as pool:
        main(run_name=args.run_name, pool=pool, overwrite=args.overwrite)

    sys.exit(0)
