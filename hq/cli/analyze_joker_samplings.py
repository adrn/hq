# Standard library
import os

# Third-party
import astropy.units as u
from astropy.table import QTable, vstack, join
import h5py
import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
import thejoker as tj
from thejoker.multiproc_helpers import batch_tasks
from thejoker.samples_analysis import (is_P_unimodal, max_phase_gap,
                                       phase_coverage, periods_spanned,
                                       phase_coverage_per_period)

# Project
from hq.log import logger
from hq.config import Config
from hq.samples_analysis import constant_model_evidence


def worker(task):
    source_ids, worker_id, c = task

    logger.debug(
        f"Worker {worker_id}: {len(source_ids)} stars left to process")

    rows = []
    units = None
    for source_id in source_ids:
        with h5py.File(c.tasks_path, 'r') as tasks_f:
            data = tj.RVData.from_timeseries(tasks_f[source_id])

        with h5py.File(c.joker_results_path, 'r') as results_f:
            if source_id not in results_f:
                logger.warning(f"No samples for: {source_id}")
                return None, None

            # Load samples from The Joker and probabilities
            samples = tj.JokerSamples.read(results_f[source_id])

        if len(samples) < 1:
            logger.warning(f"No samples for: {source_id}")
            return None, None

        row = dict()
        row[c.source_id_colname] = source_id
        row['n_visits'] = len(data)

        MAP_idx = (samples['ln_prior'] + samples['ln_likelihood']).argmax()
        MAP_sample = samples[MAP_idx]
        for k in MAP_sample.par_names:
            row['MAP_'+k] = MAP_sample[k]
        row['t_ref_bmjd'] = MAP_sample.t_ref.tcb.mjd
        row['MAP_t0_bmjd'] = MAP_sample.get_t0().tcb.mjd

        row['MAP_ln_likelihood'] = samples['ln_likelihood'][MAP_idx]
        row['MAP_ln_prior'] = samples['ln_prior'][MAP_idx]

        if len(samples) == c.requested_samples_per_star:
            row['joker_completed'] = True
        else:
            row['joker_completed'] = False

        if is_P_unimodal(samples, data):
            row['unimodal'] = True
        else:
            row['unimodal'] = False

        row['baseline'] = (data.t.mjd.max() - data.t.mjd.min()) * u.day
        row['max_phase_gap'] = max_phase_gap(MAP_sample, data)
        row['phase_coverage'] = phase_coverage(MAP_sample, data)
        row['periods_spanned'] = periods_spanned(MAP_sample, data)
        row['phase_coverage_per_period'] = phase_coverage_per_period(
            MAP_sample, data)

        # Use the max marginal likelihood sample
        lls = samples.ln_unmarginalized_likelihood(data)
        row['max_unmarginalized_ln_likelihood'] = lls.max()

        # Compute the evidence p(D) for the Kepler model, and constant RV model
        row['constant_ln_evidence'] = constant_model_evidence(data)
        row['kepler_ln_evidence'] = (logsumexp(samples['ln_likelihood']
                                               + samples['ln_prior'])
                                     - np.log(len(samples)))

        if units is None:
            units = dict()
            for k in row.keys():
                if hasattr(row[k], 'unit'):
                    units[k] = row[k].unit

        for k in units:
            row[k] = row[k].value

        rows.append(row)

    tbl = QTable(rows)
    for k in units:
        tbl[k] = tbl[k] * units[k]

    return tbl


def analyze_joker_samplings(run_path, pool):
    c = Config(run_path / 'config.yml')

    # numbers we need to validate
    for path in [c.joker_results_path, c.tasks_path]:
        if not os.path.exists(path):
            raise IOError(f"File {path} does not exist! Did you run the "
                          "preceding pipeline steps?")

    # Get data files out of config file:
    logger.debug("Loading data...")
    source_ids = np.unique(c.data[c.source_id_colname])
    tasks = batch_tasks(len(source_ids), pool.size, arr=source_ids, args=(c, ))

    logger.info(f'Done preparing tasks: {len(tasks)} stars in process queue')

    sub_tbls = []
    for tbl in tqdm(pool.map(worker, tasks), total=len(tasks)):
        if tbl is not None:
            sub_tbls.append(tbl)

    tbl = vstack(sub_tbls)

    # load results from running run_fit_constant.py:
    constant_tbl = QTable.read(c.cache_path / 'constant.fits')
    tbl = join(tbl, constant_tbl, keys=c.source_id_colname)

    tbl.write(c.metadata_joker_path, overwrite=True)
