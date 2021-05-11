# Standard library
import time

# Third-party
import astropy.table as at
import numpy as np
import pymc3 as pm
import thejoker as tj
import pymc3_ext as pmx
from thejoker.samples_helpers import inferencedata_to_samples

# Project
from hq.log import logger
from hq.samples_analysis import extract_MAP_sample
from hq.config import Config


def run_mcmc(run_path, index, seed=None, overwrite=False):
    # Load the analyzed joker samplings file, only keep unimodal:
    c = Config(run_path / 'config.yml')

    joker_metadata = at.QTable.read(c.metadata_joker_file)
    unimodal_tbl = joker_metadata[joker_metadata['unimodal']]

    if index > len(unimodal_tbl)-1:
        raise ValueError("Index is larger than the number of unimodal sources")

    metadata_row = unimodal_tbl[index]
    source_id = metadata_row[c.source_id_colname]

    # Read MAP sample:
    MAP_sample = extract_MAP_sample(metadata_row)
    logger.log(1, f"{source_id}: MAP sample loaded")

    prior, model = c.get_prior('mcmc')
    fixed_s_prior, fixed_s_model = c.get_prior('mcmc', fixed_s=MAP_sample['s'])

    mcmc_cache_path = c.cache_path / 'mcmc'
    mcmc_cache_path.mkdir(exist_ok=True)

    this_cache_path = mcmc_cache_path / source_id
    if this_cache_path.exists() and not overwrite:
        logger.info(f"{source_id} already done!")
        return

    # Set up The Joker:
    joker = tj.TheJoker(prior)
    fixed_s_joker = tj.TheJoker(fixed_s_prior)

    # Load the data:
    logger.debug(f"{source_id}: Loading all data")
    data = c.get_source_data(source_id)

    t0 = time.time()

    # Run MCMC:
    with fixed_s_model:
        logger.log(1, f"{source_id}: Setting up fixed s MCMC...")
        mcmc_init = fixed_s_joker.setup_mcmc(data, MAP_sample)

        if 'logp' not in fixed_s_model.named_vars:
            pm.Deterministic('logp', fixed_s_model.logpt)

        logger.debug(f"{source_id}: Starting MCMC sampling")
        trace = pmx.sample(start=mcmc_init, chains=2, cores=1,
                           tune=c.tune, draws=c.draws,
                           return_inferencedata=True,
                           random_seed=seed)

    init_samples = inferencedata_to_samples(fixed_s_prior, trace, data)
    df = trace.posterior.to_dataframe()
    mcmc_MAP_sample = init_samples[df.logp.argmax()]

    with model:
        logger.log(1, f"{source_id}: Setting up MCMC...")
        mcmc_init = joker.setup_mcmc(data, mcmc_MAP_sample)
        logger.log(1, f"{source_id}: ...setup complete")

        # HACK:
        mcmc_init['lnP'] = np.log(mcmc_init.get('P', 1.))

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
        trace = pmx.sample(start=mcmc_init, chains=4, cores=1,
                           tune=c.tune, draws=c.draws,
                           return_inferencedata=True,
                           discard_tuned_samples=False,
                           random_seed=seed)

    trace.to_netcdf(this_cache_path / 'samples.nc')
    logger.debug(f"{source_id}: Finished MCMC sampling "
                 f"({time.time()-t0:.2f} seconds)")
