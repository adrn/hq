# Standard library
import time

# Third-party
import astropy.table as at
import pymc3 as pm
import thejoker as tj
import exoplanet as xo

# Project
from hq.log import logger
from hq.samples_analysis import extract_MAP_sample
from hq.config import Config


def run_mcmc(run_path, index, overwrite=False):
    # Load the analyzed joker samplings file, only keep unimodal:
    c = Config(run_path / 'config.yml')

    joker_metadata = at.QTable.read(c.metadata_joker_path)
    unimodal_tbl = joker_metadata[joker_metadata['unimodal']]

    if index > len(unimodal_tbl)-1:
        raise ValueError("Index is larger than the number of unimodal sources")

    metadata_row = unimodal_tbl[index]

    prior = c.get_prior('mcmc')

    mcmc_cache_path = c.cache_path / 'mcmc'
    mcmc_cache_path.mkdir(exist_ok=True)

    source_id = metadata_row[c.source_id_colname]
    this_cache_path = mcmc_cache_path / source_id
    if this_cache_path.exists() and not overwrite:
        logger.info(f"{source_id} already done!")
        return

    # Set up The Joker:
    joker = tj.TheJoker(prior)

    # Load the data:
    logger.debug(f"{source_id}: Loading all data")
    data = c.get_source_data(source_id)

    t0 = time.time()

    # Read MAP sample:
    MAP_sample = extract_MAP_sample(metadata_row)
    logger.log(1, f"{source_id}: MAP sample loaded")

    # Run MCMC:
    with joker.prior.model as model:
        logger.log(1, f"{source_id}: Setting up MCMC...")
        mcmc_init = joker.setup_mcmc(data, MAP_sample)
        logger.log(1, f"{source_id}: ...setup complete")

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
            logger.log(1, f"{source_id}: setting up ln_prior in pymc3 model")

        if 'logp' not in model.named_vars:
            pm.Deterministic('logp', model.logpt)
            logger.log(1, f"{source_id}: setting up logp in pymc3 model")

        logger.debug(f"{source_id}: Starting MCMC sampling")
        trace = pm.sample(start=mcmc_init, chains=4, cores=1,
                          step=xo.get_dense_nuts_step(target_accept=0.95),
                          tune=c.tune, draws=c.draws)

    pm.save_trace(trace, directory=this_cache_path, overwrite=True)
    logger.debug(f"{source_id}: Finished MCMC sampling "
                 f"({time.time()-t0:.2f} seconds)")
