# Standard library
import os
import shutil
import sys
import time

try:
    # Hack to uniquify the theano cache path
    arg_i = sys.argv.index('--num') + 1  # index of actual number
    num = int(sys.argv[arg_i])
    theano_path = f"/tmp/theano_cache/mcmc{num}"
except ValueError:  # --num not passed
    theano_path = "/tmp/theano_cache/mcmc"

if os.path.exists(theano_path):
    shutil.rmtree(theano_path)
os.makedirs(theano_path)
os.environ["THEANO_FLAGS"] = f"base_compiledir={theano_path}"
print(f"Theano flags set to: " + os.environ['THEANO_FLAGS'])


# Third-party
from astropy.table import QTable
import numpy as np

import pymc3 as pm
import thejoker as tj
import exoplanet as xo

# Project
from hq.data import get_rvdata
from hq.log import logger
from hq.config import Config
from hq.script_helpers import get_parser
from hq.samples_analysis import extract_MAP_sample


def main(c, prior, metadata_row, overwrite=False):
    mcmc_cache_path = os.path.join(c.run_path, 'mcmc')
    os.makedirs(mcmc_cache_path, exist_ok=True)

    apogee_id = metadata_row['APOGEE_ID']

    this_cache_path = os.path.join(mcmc_cache_path, apogee_id)
    if os.path.exists(this_cache_path) and not overwrite:
        logger.info(f"{apogee_id} already done!")
        # Assume it's already done
        return

    # Set up The Joker:
    joker = tj.TheJoker(prior)

    # Load the data:
    logger.debug(f"{apogee_id}: Loading all data")
    allstar, allvisit = c.load_alldata()
    allstar = allstar[np.isin(allstar['APOGEE_ID'].astype(str), apogee_id)]
    allvisit = allvisit[np.isin(allvisit['APOGEE_ID'].astype(str),
                                allstar['APOGEE_ID'].astype(str))]
    visits = allvisit[allvisit['APOGEE_ID'] == apogee_id]
    data = get_rvdata(visits)

    t0 = time.time()

    # Read MAP sample:
    MAP_sample = extract_MAP_sample(metadata_row)
    logger.log(1, f"{apogee_id}: MAP sample loaded")

    # Run MCMC:
    with joker.prior.model as model:
        logger.log(1, f"{apogee_id}: Setting up MCMC...")
        mcmc_init = joker.setup_mcmc(data, MAP_sample)
        logger.log(1, f"{apogee_id}: ...setup complete")

        if 'ln_prior' not in model.named_vars:
            ln_prior_var = None
            for k in joker.prior._nonlinear_equiv_units:
                var = model.named_vars[k]
                try:
                    if ln_prior_var is None:
                        ln_prior_var = var.distribution.logp(var)
                    else:
                        ln_prior_var = ln_prior_var + var.distribution.logp(var)
                except Exception as e:
                    logger.warning("Cannot auto-compute log-prior value for "
                                   f"parameter {var}.")
                    print(e)
                    continue

            pm.Deterministic('ln_prior', ln_prior_var)
            logger.log(1, f"{apogee_id}: setting up ln_prior in pymc3 model")

        if 'logp' not in model.named_vars:
            pm.Deterministic('logp', model.logpt)
            logger.log(1, f"{apogee_id}: setting up logp in pymc3 model")

        logger.debug(f"{apogee_id}: Starting MCMC sampling")
        trace = pm.sample(start=mcmc_init, chains=4, cores=1,
                          step=xo.get_dense_nuts_step(target_accept=0.95),
                          tune=c.tune, draws=c.draws)

    pm.save_trace(trace, directory=this_cache_path, overwrite=True)
    logger.debug("{apogee_id}: Finished MCMC sampling ({time:.2f} seconds)"
                 .format(apogee_id=apogee_id, time=time.time() - t0))


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='Run The Joker on APOGEE data',
                        loggers=[logger])

    parser.add_argument("--num", type=int, default=None)

    args = parser.parse_args()

    # Load the analyzed joker samplings file, only keep unimodal:
    c = Config.from_run_name(args.run_name)
    prior = c.get_prior('mcmc')

    joker_metadata = QTable.read(c.metadata_joker_path)
    unimodal_tbl = joker_metadata[joker_metadata['unimodal']]
    if args.num is None:
        print(f"{len(unimodal_tbl)} rows in unimodal table - exiting...")
        sys.exit(0)

    metadata_row = unimodal_tbl[args.num]

    main(c, prior,
         metadata_row=metadata_row,
         overwrite=args.overwrite)

    sys.exit(0)
