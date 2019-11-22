# Standard library
from os import path
import sys

# Third-party
from astropy.table import Table
import h5py
import numpy as np
from thejoker.logging import log as joker_logger
from tqdm import tqdm
from schwimmbad import SerialPool

# Project
from hq.data import get_rvdata
from hq.log import logger
from hq.config import Config
from hq.script_helpers import get_parser


def main(run_name, pool, overwrite=False):
    c = Config.from_name(run_name)

    if path.exists(c.tasks_path) and not overwrite:
        logger.info(f"File {c.tasks_path} already exists: Use -o/--overwrite if"
                    " needed")
        return

    # Load the full allvisit file, but only some columns:
    allvisit = Table()
    for k in ['APOGEE_ID', 'JD', 'VHELIO', 'VRELERR', 'SNR']:
        allvisit[k] = c.allvisit[k]

    logger.debug("Loading data and preparing tasks...")
    apogee_ids = np.unique(c.allstar['APOGEE_ID'])
    with h5py.File(c.tasks_path, 'w') as f:
        for apogee_id in tqdm(apogee_ids):
            visits = allvisit[allvisit['APOGEE_ID'] == apogee_id]
            data = get_rvdata(visits)

            g = f.create_group(apogee_id)
            data.to_hdf5(g)

    logger.info('Done preparing tasks: {0} stars in process queue'
                .format(len(apogee_ids)))


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='Run The Joker on APOGEE data',
                        loggers=[logger, joker_logger])

    args = parser.parse_args()

    Pool = SerialPool

    with Pool() as pool:
        main(run_name=args.run_name, pool=pool, overwrite=args.overwrite)

    sys.exit(0)
