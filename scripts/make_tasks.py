# Standard library
from os import path
import pickle
import sys

# Third-party
import numpy as np
from thejoker.log import log as joker_logger
from tqdm import tqdm
import yaml
from schwimmbad import SerialPool

# Project
from hq.data import get_rvdata
from hq.log import logger
from hq.config import HQ_CACHE_PATH, config_to_alldata
from hq.script_helpers import get_parser


def main(run_name, pool, overwrite=False, seed=None):
    run_path = path.join(HQ_CACHE_PATH, run_name)
    with open(path.join(run_path, 'config.yml'), 'r') as f:
        config = yaml.load(f.read())

    tasks_path = path.join(HQ_CACHE_PATH, run_name, 'tmp-tasks.pkl')

    if path.exists(tasks_path) and not overwrite:
        logger.info("File {} already exists. Use --overwrite if needed"
                    .format(tasks_path))
        return

    # Get data files out of config file:
    allstar, allvisit = config_to_alldata(config)
    allvisit = allvisit[np.isin(allvisit['APOGEE_ID'], allstar['APOGEE_ID'])]

    tasks = []
    logger.debug("Loading data and preparing tasks...")
    for star in tqdm(allstar):
        visits = allvisit[allvisit['APOGEE_ID'] == star['APOGEE_ID']]
        data = get_rvdata(visits)
        tasks.append([star['APOGEE_ID'], data])

    logger.info('Done preparing tasks: {0} stars in process queue'
                .format(len(tasks)))

    with open(tasks_path, 'wb') as f:
        pickle.dump(tasks, f)


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='Run The Joker on APOGEE data',
                        loggers=[logger, joker_logger])

    parser.add_argument("--name", dest="run_name", required=True,
                        type=str, help="The name of the run.")

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true",
                        help="Overwrite any existing samplings")

    args = parser.parse_args()

    Pool = SerialPool

    with Pool() as pool:
        main(run_name=args.run_name, pool=pool, overwrite=args.overwrite,
             seed=args.seed)

    sys.exit(0)
