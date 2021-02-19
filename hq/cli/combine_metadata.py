# Third-party
import numpy as np
import theano
theano.config.optimizer = 'None'
theano.config.mode = 'FAST_COMPILE'
theano.config.reoptimize_unpickled_function = False
theano.config.cxx = ""
import astropy.table as at

# Project
from hq.config import Config
from hq.log import logger


def combine_metadata(run_path, overwrite=False):
    c = Config(run_path / 'config.yml')

    if c.metadata_file.exists() and not overwrite:
        logger.info(f"metadata file already exists {str(c.metadata_file)}")
        return

    meta = at.Table.read(c.metadata_joker_file)
    mcmc_meta = at.Table.read(c.metadata_mcmc_file)
    constant = at.Table.read(c.constant_results_file)

    final_colnames = [
        c.source_id_colname,
        'n_visits',
        'MAP_P',
        'MAP_P_err',
        'MAP_e',
        'MAP_e_err',
        'MAP_omega',
        'MAP_omega_err',
        'MAP_M0',
        'MAP_M0_err',
        'MAP_K',
        'MAP_K_err',
        'MAP_v0',
        'MAP_v0_err',
        'MAP_s',
        'MAP_s_err',
        'MAP_t0_bmjd',
        't_ref_bmjd',
        'baseline',
        'MAP_ln_likelihood',
        'MAP_ln_prior',
        'max_unmarginalized_ln_likelihood',
        'max_phase_gap',
        'periods_spanned',
        'phase_coverage',
        'phase_coverage_per_period',
        'unimodal',
        'joker_completed',
        'mcmc_completed',
        'mcmc_success',
        'gelman_rubin_max',
        'constant_ln_likelihood',
        'robust_constant_ln_likelihood']

    meta = at.join(meta, constant, keys=c.source_id_colname, join_type='left')
    master = at.join(meta, mcmc_meta, keys=c.source_id_colname,
                     join_type='left',
                     uniq_col_name="{table_name}{col_name}",
                     table_names=["", "mcmc_"])
    master = at.unique(master, keys=c.source_id_colname)

    master['mcmc_completed'] = master['mcmc_completed'].filled(False)
    master['mcmc_success'] = master['mcmc_success'].filled(False)
    if hasattr(master['joker_completed'], 'filled'):
        master['joker_completed'] = master['joker_completed'].filled(False)

    for colname in mcmc_meta.colnames:
        if colname == c.source_id_colname:
            continue

        mcmc_colname = f'mcmc_{colname}'
        if mcmc_colname not in master.colnames:
            mcmc_colname = colname

        print(f"Filling {colname} with {mcmc_colname}")
        master[colname][master['mcmc_success']] = \
            master[mcmc_colname][master['mcmc_success']]

    master = master[final_colnames]
    master = at.QTable(master)

    for col in master.colnames:
        if col.endswith('_err'):
            master[col][~master['mcmc_completed']] = np.nan

    # load results from running run_fit_constant.py:
    constant_tbl = at.QTable.read(c.constant_results_file)
    master = at.join(master, constant_tbl, keys=c.source_id_colname)
    master = at.unique(master, keys=c.source_id_colname)

    master.write(c.metadata_file, overwrite=True)
