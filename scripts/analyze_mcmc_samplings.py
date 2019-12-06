# Standard library
import os
import sys

# Third-party
import theano
theano.config.optimizer = 'None'
theano.config.mode = 'FAST_COMPILE'
theano.config.reoptimize_unpickled_function = False
theano.config.cxx = ""
import astropy.units as u
from astropy.stats import median_absolute_deviation
from astropy.table import Table, QTable, vstack
import h5py
import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
import thejoker as tj
from thejoker.multiproc_helpers import batch_tasks
import pymc3 as pm

# Project
from hq.log import logger
from hq.config import Config
from hq.samples_analysis import (unimodal_P, max_phase_gap, phase_coverage,
                                 periods_spanned, phase_coverage_per_period,
                                 constant_model_evidence)

from run_fit_constant import ln_normal


def worker(task):
    apogee_ids, worker_id, c = task

    logger.debug(f"Worker {worker_id}: {len(apogee_ids)} stars left to process")
    prior = c.get_prior()

    rows = []
    sub_samples = {}
    units = None
    for apogee_id in apogee_ids:
        with h5py.File(c.tasks_path, 'r') as tasks_f:
            data = tj.RVData.from_timeseries(tasks_f[apogee_id])

        this_mcmc_path = os.path.join(c.run_path, 'mcmc', apogee_id)
        if not os.path.exists(this_mcmc_path):
            logger.debug(f"{apogee_id}: MCMC path does not exist at "
                         f"{this_mcmc_path}")

        joker = tj.TheJoker(prior)
        with prior.model:
            trace = pm.load_trace(this_mcmc_path)

        for i in trace.chains:
            trace._straces[i].varnames.append('ln_prior')
            trace._straces[i].varnames.append('logp')

        samples = joker.trace_to_samples(trace, data)
        samples['ln_prior'] = trace['ln_prior']

        # TODO: re-enable likelihood calculation
        # with prior.model as model:
        #     _ = joker.setup_mcmc(data, samples[0])  # add Kepler model...

        #     ln_likelihoods = []
        #     for chain in trace.points():
        #         ll = model.observed_RVs[0].logp_elemwise(chain)
        #         ln_likelihoods.append(ll)
        # ln_likelihood = np.array(ln_likelihoods).sum(axis=1)
        # HACK:
        ln_likelihood = trace['logp'] - samples['ln_prior']

        row = dict()
        row['APOGEE_ID'] = apogee_id
        row['n_visits'] = len(data)

        MAP_idx = (samples['ln_prior'] + ln_likelihood).argmax()
        MAP_sample = samples[MAP_idx]
        for k in MAP_sample.par_names:
            row[f'MAP_{k}'] = MAP_sample[k]

        err = dict()
        for k in samples.par_names:
            err[k] = 1.5 * median_absolute_deviation(samples[k])
            row[f'MAP_{k}_err'] = err[k]

        row['t0_bmjd'] = MAP_sample.t0.tcb.mjd

        row['MAP_ln_likelihood'] = ln_likelihood[MAP_idx]
        row['MAP_ln_prior'] = samples['ln_prior'][MAP_idx]

        row['joker_completed'] = False
        row['mcmc_completed'] = True

        if unimodal_P(samples, data):
            row['unimodal'] = True
        else:
            row['unimodal'] = False
        
        stat_df = pm.summary(trace)
        row['gelman_rubin_max'] = stat_df['r_hat'].max()
        row['mcmc_success'] = row['gelman_rubin_max'] <= 1.1

        row['baseline'] = (data.t.mjd.max() - data.t.mjd.min()) * u.day

        if row['mcmc_success']:
            row['max_phase_gap'] = max_phase_gap(MAP_sample, data)
            row['phase_coverage'] = phase_coverage(MAP_sample, data)
            row['periods_spanned'] = periods_spanned(MAP_sample, data)
            row['phase_coverage_per_period'] = phase_coverage_per_period(MAP_sample,
                                                                         data)
        else:
            row['max_phase_gap'] = np.nan * u.one
            row['phase_coverage'] = np.nan * u.one
            row['periods_spanned'] = np.nan * u.one
            row['phase_coverage_per_period'] = np.nan * u.one

        # Use the max marginal likelihood sample
        _unit = data.rv.unit
        max_ll_sample = samples[ln_likelihood.argmax()]
        orbit = max_ll_sample.get_orbit()
        var = data.rv_err**2 + max_ll_sample['s']**2
        ll = ln_normal(orbit.radial_velocity(data.t).to_value(_unit),
                       data.rv.to_value(_unit),
                       var.to_value(_unit**2)).sum()
        row['max_unmarginalized_ln_likelihood'] = ll

        # Compute the evidence p(D) for the Kepler model and for the constant RV
        row['constant_ln_evidence'] = constant_model_evidence(data)
        row['kepler_ln_evidence'] = (logsumexp(ln_likelihood +
                                               samples['ln_prior'])
                                     - np.log(len(samples)))

        if units is None:
            units = dict()
            for k in row.keys():
                if hasattr(row[k], 'unit'):
                    units[k] = row[k].unit

        for k in units:
            if hasattr(row[k], 'unit'):
                row[k] = row[k].value

        rows.append(row)

        # Now write out the requested number of samples:
        idx = np.random.choice(len(samples), size=c.requested_samples_per_star,
                               replace=False)
        sub_samples[apogee_id] = samples[idx]
        sub_samples[apogee_id]['ln_prior'] = samples['ln_prior'][idx]
        sub_samples[apogee_id]['ln_likelihood'] = ln_likelihood[idx]

    tbl = Table(rows)
    return {'tbl': tbl, 'samples': sub_samples, 'units': units}


def main(run_name, pool):
    c = Config.from_run_name(run_name)

    apogee_ids = sorted([x for x in os.listdir(os.path.join(c.run_path, 'mcmc'))
                         if not x.startswith('.')])

    tasks = batch_tasks(len(apogee_ids), pool.size, arr=apogee_ids, args=(c, ))
    logger.info(f'Done preparing tasks: {len(tasks)} stars in process queue')

    sub_tbls = []
    all_samples = {}
    for result in tqdm(pool.map(worker, tasks), total=len(tasks)):
        if result is not None:
            sub_tbls.append(result['tbl'])
            all_samples.update(result['samples'])

    # Write the MCMC metadata table
    tbl = vstack(sub_tbls)
    for k in result['units']:
        tbl[k].unit = result['units'][k]
    tbl = QTable(tbl)
    tbl.write(c.metadata_mcmc_path, overwrite=True)

    # Now write out all of the individual samplings:
    with h5py.File(c.mcmc_results_path, 'a') as results_f:
        for apogee_id, samples in all_samples.items():
            if apogee_id in results_f:
                del results_f[apogee_id]
            g = results_f.create_group(apogee_id)
            samples.write(g)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    from hq.script_helpers import get_parser

    # Define parser object
    parser = get_parser(description='TODO',
                        loggers=logger)

    args = parser.parse_args()

    # with threadpool_limits(limits=1, user_api='blas'):
    with args.Pool(**args.Pool_kwargs) as pool:
        main(run_name=args.run_name, pool=pool)

    sys.exit(0)
