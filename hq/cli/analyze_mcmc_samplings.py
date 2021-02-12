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
from astropy.table import Table, QTable, vstack
import h5py
import numpy as np
from tqdm import tqdm
import thejoker as tj
from thejoker.multiproc_helpers import batch_tasks
import pymc3 as pm
import arviz as az

# Project
from hq.log import logger
from hq.config import Config
from .analyze_joker_samplings import compute_metadata


def worker(task):
    source_ids, worker_id, c = task

    logger.debug(f"Worker {worker_id}: {len(source_ids)} stars left to process")
    prior = c.get_prior()

    rows = []
    sub_samples = {}
    units = None
    for source_id in source_ids:
        with h5py.File(c.tasks_path, 'r') as tasks_f:
            data = tj.RVData.from_timeseries(tasks_f[source_id])

        this_mcmc_path = os.path.join(c.cache_path, 'mcmc', source_id)
        if not os.path.exists(this_mcmc_path):
            logger.debug(f"{source_id}: MCMC path does not exist at "
                         f"{this_mcmc_path}")

        joker = tj.TheJoker(prior)
        with prior.model:
            trace = pm.load_trace(this_mcmc_path)

        for i in trace.chains:
            trace._straces[i].varnames.append('ln_prior')
            trace._straces[i].varnames.append('logp')

        samples = joker.trace_to_samples(trace, data)
        samples['ln_prior'] = trace['ln_prior']
        # HACK:
        samples['ln_likelihood'] = trace['logp'] - samples['ln_prior']

        row = compute_metadata(c, samples, data)

        row['joker_completed'] = False
        row['mcmc_completed'] = True

        stat_df = az.summary(trace)
        row['gelman_rubin_max'] = stat_df['r_hat'].max()
        row['mcmc_success'] = row['gelman_rubin_max'] <= 1.1

        if not row['mcmc_success']:
            row['max_phase_gap'] = np.nan * u.one
            row['phase_coverage'] = np.nan * u.one
            row['periods_spanned'] = np.nan * u.one
            row['phase_coverage_per_period'] = np.nan * u.one

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
        sub_samples[source_id] = samples[idx]
        sub_samples[source_id]['ln_prior'] = samples['ln_prior'][idx]

        sub_samples[source_id]['ln_likelihood'] = samples['ln_likelihood'][idx]

    tbl = Table(rows)
    return {'tbl': tbl, 'samples': sub_samples, 'units': units}


def analyze_mcmc_samplings(run_path, pool):
    c = Config(run_path / 'config.yml')

    source_ids = sorted([x for x in os.listdir(c.cache_path / 'mcmc')
                         if not x.startswith('.')])

    tasks = batch_tasks(len(source_ids), pool.size, arr=source_ids, args=(c, ))
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
        for source_id, samples in all_samples.items():
            if source_id in results_f:
                del results_f[source_id]
            g = results_f.create_group(source_id)
            samples.write(g)
