# Standard library
import os
import sys

# Third-party
import astropy.units as u
from astropy.table import Table, vstack, join
import h5py
import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
import thejoker as tj
from thejoker.multiproc_helpers import batch_tasks

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

    rows = []
    for apogee_id in apogee_ids:
        with h5py.File(c.tasks_path, 'r') as tasks_f:
            data = tj.RVData.from_timeseries(tasks_f[apogee_id])

        with h5py.File(c.joker_results_path, 'r') as results_f:
            if apogee_id not in results_f:
                logger.warning("No samples for: {}".format(apogee_id))
                return None, None

            # Load samples from The Joker and probabilities
            samples = tj.JokerSamples.read(results_f[apogee_id])

        if len(samples) < 1:
            logger.warning("No samples for: {}".format(apogee_id))
            return None, None

        row = dict()
        row['APOGEE_ID'] = apogee_id
        row['n_visits'] = len(data)

        MAP_idx = (samples['ln_prior'] + samples['ln_likelihood']).argmax()
        MAP_sample = samples[MAP_idx]
        for k in MAP_sample.par_names:
            row['MAP_'+k] = MAP_sample[k]
        row['t0_bmjd'] = MAP_sample.t0.tcb.mjd

        row['MAP_ln_likelihood'] = samples['ln_likelihood'][MAP_idx]
        row['MAP_ln_prior'] = samples['ln_prior'][MAP_idx]

        if len(samples) == c.requested_samples_per_star:
            row['joker_completed'] = True
        else:
            row['joker_completed'] = False

        if unimodal_P(samples, data):
            row['unimodal'] = True
        else:
            row['unimodal'] = False

        row['baseline'] = (data.t.mjd.max() - data.t.mjd.min()) * u.day
        row['max_phase_gap'] = max_phase_gap(MAP_sample, data)
        row['phase_coverage'] = phase_coverage(MAP_sample, data)
        row['periods_spanned'] = periods_spanned(MAP_sample, data)
        row['phase_coverage_per_period'] = phase_coverage_per_period(MAP_sample,
                                                                     data)

        # Use the max marginal likelihood sample
        _unit = data.rv.unit
        max_ll_sample = samples[samples['ln_likelihood'].argmax()]
        orbit = max_ll_sample.get_orbit()
        var = data.rv_err**2 + max_ll_sample['s']**2
        ll = ln_normal(orbit.radial_velocity(data.t).to_value(_unit),
                       data.rv.to_value(_unit),
                       var.to_value(_unit**2)).sum()
        row['max_unmarginalized_ln_likelihood'] = ll

        # Compute the evidence p(D) for the Kepler model and for the constant RV
        row['constant_ln_evidence'] = constant_model_evidence(data)
        row['kepler_ln_evidence'] = (logsumexp(samples['ln_likelihood']
                                               + samples['ln_prior'])
                                     - np.log(len(samples)))

        rows.append(row)

    units = dict()
    for k in row:
        if hasattr(row[k], 'unit'):
            units[k] = row[k].unit
            row[k] = row[k].value

    return row, units


def main(run_name, pool):
    c = Config.from_run_name(run_name)

    # numbers we need to validate
    for path in [c.joker_results_path, c.tasks_path]:
        if not os.path.exists(path):
            raise IOError(f"File {path} does not exist! Did you run the "
                          "preceding pipeline steps?")

    # Get data files out of config file:
    logger.debug("Loading data...")
    allstar, _ = c.load_alldata()
    apogee_ids = np.unique(allstar['APOGEE_ID'])
    tasks = batch_tasks(len(apogee_ids), pool.size, arr=apogee_ids, args=(c, ))

    logger.info(f'Done preparing tasks: {len(tasks)} stars in process queue')

    sub_tbls = []
    for tbl, units in tqdm(pool.map(worker, tasks), total=len(tasks)):
        if tbl is not None:
            sub_tbls.append(tbl)

    tbl = vstack(sub_tbls)

    for k, unit in units.items():
        tbl[k] = tbl[k] * unit

    # load results from running run_fit_constant.py:
    constant_path = os.path.join(c.run_path, 'constant.fits')
    constant_tbl = Table.read(constant_path)
    tbl = join(tbl, constant_tbl, keys='APOGEE_ID')

    tbl.write(c.metadata_path, overwrite=True)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    from hq.script_helpers import get_parser

    # Define parser object
    parser = get_parser(description='TODO',
                        loggers=logger)

    args = parser.parse_args()

    with threadpool_limits(limits=1, user_api='blas'):
        with args.Pool(**args.Pool_kwargs) as pool:
            main(run_name=args.run_name, pool=pool)

    sys.exit(0)
