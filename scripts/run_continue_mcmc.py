# Standard library
import os
import sys
import time

# Third-party
from astropy.table import QTable
import numpy as np
from thejoker.logging import logger as joker_logger
import thejoker as tj
import pymc3 as pm
import exoplanet as xo

# Project
from hq.data import get_rvdata
from hq.log import logger
from hq.config import Config
from hq.script_helpers import get_parser
from hq.samples_analysis import extract_MAP_sample


def main(c, metadata_row, overwrite=False):
    mcmc_cache_path = os.path.join(c.run_path, 'mcmc')
    os.makedirs(mcmc_cache_path, exist_ok=True)

    apogee_id = metadata_row['APOGEE_ID']

    this_cache_path = os.path.join(mcmc_cache_path, apogee_id)
    if os.path.exists(this_cache_path) and not overwrite:
        # Assume it's already done
        return

    # Hack to uniquify the theano cache path
    _path = os.path.join(c.run_path, "theano_cache", f"p{apogee_id}")
    os.makedirs(_path, exist_ok=True)
    os.environ["THEANO_FLAGS"] = f"base_compiledir={_path}"

    # Set up The Joker:
    joker = tj.TheJoker(c.get_prior())

    # Load the data:
    allstar, allvisit = c.load_alldata()
    allstar = allstar[np.isin(allstar['APOGEE_ID'].astype(str), apogee_id)]
    allvisit = allvisit[np.isin(allvisit['APOGEE_ID'].astype(str),
                                allstar['APOGEE_ID'].astype(str))]
    visits = allvisit[allvisit['APOGEE_ID'] == apogee_id]
    data = get_rvdata(visits)

    t0 = time.time()
    logger.debug(f"{apogee_id}: Starting MCMC sampling")

    # Read MAP sample:
    MAP_sample = extract_MAP_sample(metadata_row)

    # Run MCMC:
    with joker.prior.model:
        mcmc_init = joker.setup_mcmc(data, MAP_sample)
        trace = pm.sample(start=mcmc_init, chains=4, cores=1,
                          step=xo.get_dense_nuts_step(target_accept=0.95),
                          tune=c.tune, draws=c.draws)

    pm.save_trace(trace, directory=this_cache_path)
    logger.debug("{apogee_id}: Finished MCMC sampling ({time:.2f} seconds)"
                 .format(apogee_id=apogee_id, time=time.time() - t0))


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits

    # Define parser object
    parser = get_parser(description='Run The Joker on APOGEE data',
                        loggers=[logger, joker_logger])

    parser.add_argument("--num", type=int, default=None)

    args = parser.parse_args()

    # Load the analyzed joker samplings file, only keep unimodal:
    c = Config.from_run_name(args.run_name)
    joker_metadata = QTable.read(c.metadata_path)
    unimodal_tbl = joker_metadata[joker_metadata['unimodal'] &
                                  (joker_metadata['periods_spanned'] > 1.)]
    if args.num is None:
        print(f"{len(unimodal_tbl)} rows in unimodal table - exiting...")
        sys.exit(0)

    metadata_row = unimodal_tbl[args.num]

    with threadpool_limits(limits=1, user_api='blas'):
        main(c,
             metadata_row=metadata_row,
             overwrite=args.overwrite)

    sys.exit(0)
