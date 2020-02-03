# Standard library
import os
import sys

# Third-party
import numpy as np
import theano
theano.config.optimizer = 'None'
theano.config.mode = 'FAST_COMPILE'
theano.config.reoptimize_unpickled_function = False
theano.config.cxx = ""
from astropy.table import Table, QTable, join

# Project
from hq.config import Config
from hq.log import logger


def main(run_name, overwrite=False):
    c = Config.from_run_name(run_name)

    if os.path.exists(c.metadata_path) and not overwrite:
        logger.info(f"metadata file already exists {c.metadata_path}")
        return

    meta = Table.read(c.metadata_joker_path)
    mcmc_meta = Table.read(c.metadata_mcmc_path)

    final_colnames = [
        'APOGEE_ID',
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
        't0_bmjd',
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
        'robust_constant_ln_likelihood',
        'constant_ln_evidence',
        'kepler_ln_evidence']

    master = join(meta, mcmc_meta, keys='APOGEE_ID', join_type='left',
                  uniq_col_name="{table_name}{col_name}",
                  table_names=["", "mcmc_"])

    master['mcmc_completed'] = master['mcmc_completed'].filled(False)
    master['mcmc_success'] = master['mcmc_success'].filled(False)
    master['joker_completed'] = master['joker_completed'].filled(False)

    for colname in mcmc_meta.colnames:
        if colname == 'APOGEE_ID':
            continue

        mcmc_colname = f'mcmc_{colname}'
        if mcmc_colname not in master.colnames:
            mcmc_colname = colname

        print(f"Filling {colname} with {mcmc_colname}")
        master[colname][master['mcmc_success']] = master[mcmc_colname][master['mcmc_success']]

    master = master[final_colnames]
    master = QTable(master)

    for col in master.colnames:
        if col.endswith('_err'):
            master[col][~master['mcmc_completed']] = np.nan

    master.write(c.metadata_path, overwrite=True)


if __name__ == '__main__':
    from hq.script_helpers import get_parser

    # Define parser object
    parser = get_parser(description='TODO',
                        loggers=logger)

    args = parser.parse_args()

    main(run_name=args.run_name, overwrite=args.overwrite)

    sys.exit(0)
