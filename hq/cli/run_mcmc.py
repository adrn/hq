# Standard library
import time

# Third-party
import arviz as az
import astropy.table as at
import thejoker as tj
import pymc3_ext as pmx
from thejoker.samples_helpers import inferencedata_to_samples

# Project
from hq.log import logger
from hq.samples_analysis import extract_MAP_sample
from hq.config import Config


def run_mcmc(run_path, index, seed=None, overwrite=False):
    # Load the analyzed joker samplings file, only keep unimodal:
    conf = Config(run_path / 'config.yml')

    joker_metadata = at.QTable.read(conf.metadata_joker_file)
    unimodal_tbl = joker_metadata[joker_metadata['unimodal']]

    if index > len(unimodal_tbl) - 1:
        raise ValueError("Index is larger than the number of unimodal sources")

    metadata_row = unimodal_tbl[index]
    source_id = metadata_row[conf.source_id_colname]

    # Make sure the root mcmc path exists:
    mcmc_cache_path = conf.cache_path / 'mcmc'
    mcmc_cache_path.mkdir(exist_ok=True)

    # Read the source data and MAP sample:
    data = conf.get_source_data(source_id)
    joker_MAP_sample = extract_MAP_sample(metadata_row)
    logger.log(1, f"{source_id}: MAP sample loaded")

    this_cache_path = mcmc_cache_path / source_id
    samples_file = this_cache_path / 'samples.nc'
    if samples_file.exists() and not overwrite:
        logger.info(f"{source_id} already done!")
        return

    stat_names = ['P', 'K', 'e', 'M0', 'omega', 't_peri', 'v0']

    # -------- Initial run ---------
    # Fix the excess variance parameter to the MAP value from running The Joker
    time0 = time.time()

    fixed_s_prior, fixed_s_model = conf.get_prior(
        'mcmc',
        MAP_sample=joker_MAP_sample,
        fixed_s=joker_MAP_sample['s'])

    with fixed_s_model:
        joker = tj.TheJoker(fixed_s_prior)

        mcmc_init = joker.setup_mcmc(data, joker_MAP_sample,
                                     custom_func=conf.get_custom_init_mcmc())

        init_trace = pmx.sample(
            start=mcmc_init, chains=conf.mcmc_chains, cores=1,
            init='adapt_full',
            tune=conf.mcmc_tune_steps,
            draws=1000,  # MAGIC NUMBER - TODO: make config?
            return_inferencedata=True,
            discard_tuned_samples=True,
            random_seed=seed,
            target_accept=conf.mcmc_target_accept)

    init_samples = inferencedata_to_samples(joker.prior, init_trace, data)
    tmp_MAP_sample = init_samples[init_samples['ln_posterior'].argmax()]
    init_trace.to_netcdf(this_cache_path / 'init_samples.nc')

    stat_names.extend([x for x in mcmc_init.keys() if x not in stat_names])
    summ = az.summary(init_trace.posterior, var_names=stat_names)
    with open(this_cache_path / 'init-sample-stats.csv') as f:
        f.write(summ.to_csv())

    if summ.r_hat.max() > conf.mcmc_max_r_hat:
        pass

    # -------- Main run ---------

    prior, model = conf.get_prior(
        'mcmc',
        MAP_sample=tmp_MAP_sample)

    with model:
        joker = tj.TheJoker(prior)
        mcmc_init = joker.setup_mcmc(data, tmp_MAP_sample,
                                     custom_func=conf.get_custom_init_mcmc())

        logger.debug(f"{source_id}: Starting initial (fixed s) MCMC sampling")
        trace = pmx.sample(
            start=mcmc_init, chains=conf.mcmc_chains, cores=1,
            init='adapt_full',
            tune=conf.mcmc_tune_steps,
            draws=conf.mcmc_draw_steps,
            return_inferencedata=True,
            discard_tuned_samples=False,
            random_seed=seed,
            target_accept=conf.mcmc_target_accept)

    stat_names.extend([x for x in mcmc_init.keys() if x not in stat_names])
    summ = az.summary(trace.posterior, var_names=stat_names)
    with open(this_cache_path / 'init-sample-stats.csv') as f:
        f.write(summ.to_csv())

    trace.to_netcdf(samples_file)
    logger.debug(f"{source_id}: Finished MCMC sampling "
                 f"({time.time()-time0:.2f} seconds)")
