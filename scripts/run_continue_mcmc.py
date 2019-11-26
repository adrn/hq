# Standard library
import os
import sys
import time

# Third-party
from astropy.table import QTable
import numpy as np
from thejoker.logging import logger as joker_logger
import thejoker as tj
from tqdm import tqdm

# Project
from hq.data import get_rvdata
from hq.log import logger
from hq.config import Config
from hq.script_helpers import get_parser
from hq.samples_analysis import extract_MAP_sample


def worker(apogee_ids, data, c, MAP_sample, mcmc_cache_path, sample_kw):
    path = os.path.join(c.run_path,
                        os.path.abspath("theano_cache"),
                        f"p{apogee_ids[0]}")
    os.makedirs(path, exist_ok=True)
    os.environ["THEANO_FLAGS"] = f"base_compiledir={path}"

    import pymc3 as pm
    import exoplanet as xo

    joker = tj.TheJoker(c.get_prior())

    for apogee_id in apogee_ids:
        this_cache_path = os.path.join(mcmc_cache_path, apogee_id)
        if os.path.exists(this_cache_path):
            # Assume it's already done
            return

        t0 = time.time()
        logger.debug(f"{apogee_id}: Starting MCMC sampling")

        with joker.prior.model:
            mcmc_init = joker.setup_mcmc(data, MAP_sample)
            trace = pm.sample(start=mcmc_init, chains=4, cores=1,
                              step=xo.get_dense_nuts_step(target_accept=0.95),
                              **sample_kw)

        pm.save_trace(trace, directory=this_cache_path)
        logger.debug("{apogee_id}: Finished MCMC sampling ({time:.2f} seconds)"
                     .format(apogee_id=apogee_id, time=time.time() - t0))


def main(run_name, pool, overwrite=False):
    c = Config.from_run_name(run_name)

    mcmc_cache_path = os.path.join(c.run_path, 'mcmc')
    os.makedirs(mcmc_cache_path, exist_ok=True)

    # Load the analyzed joker samplings file, only keep unimodal:
    joker_metadata = QTable.read(c.metadata_path)
    unimodal_tbl = joker_metadata[joker_metadata['unimodal'] &
                                  (joker_metadata['periods_spanned'] > 1.)]

    # Load the data:
    allstar, allvisit = c.load_alldata()
    allstar = allstar[np.isin(allstar['APOGEE_ID'].astype(str),
                              unimodal_tbl['APOGEE_ID'].astype(str))]
    allvisit = allvisit[np.isin(allvisit['APOGEE_ID'].astype(str),
                                allstar['APOGEE_ID'].astype(str))]

    sample_kw = dict(tune=c.tune, draws=c.draws)

    tasks = []
    logger.debug("Loading data and preparing tasks...")
    for row in tqdm(unimodal_tbl):
        visits = allvisit[allvisit['APOGEE_ID'] == row['APOGEE_ID']]
        data = get_rvdata(visits)

        MAP_sample = extract_MAP_sample(row)
        tasks.append([row['APOGEE_ID'], data, c, MAP_sample,
                      mcmc_cache_path, sample_kw])

    logger.info(f'Done preparing tasks: {len(tasks)} stars in process queue')

    for r in tqdm(pool.starmap(worker, tasks), total=len(tasks)):
        pass


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='Run The Joker on APOGEE data',
                        loggers=[logger, joker_logger])

    args = parser.parse_args()

    with args.Pool(**args.Pool_kwargs) as pool:
        main(run_name=args.run_name, pool=pool, overwrite=args.overwrite)

    sys.exit(0)
