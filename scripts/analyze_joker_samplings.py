# Standard library
import os
import sys

# Third-party
import astropy.units as u
from astropy.table import Table
import h5py
import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
import thejoker as tj

# Project
from hq.log import logger
from hq.config import Config
from hq.script_helpers import get_parser
from hq.samples_analysis import (unimodal_P, max_phase_gap, phase_coverage,
                                 periods_spanned, phase_coverage_per_period,
                                 constant_model_evidence)
from run_fit_constant import ln_normal


def worker(apogee_id, data, poly_trend, n_requested_samples, results_path):
    with h5py.File(results_path, 'r') as results_f:
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

    if len(samples) == n_requested_samples:
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

    # Compute the evidence, p(D), for the Kepler model and for the constant RV
    row['constant_ln_evidence'] = constant_model_evidence(data)
    row['kepler_ln_evidence'] = (logsumexp(samples['ln_likelihood']
                                           + samples['ln_prior'])
                                 - np.log(len(samples)))

    units = dict()
    for k in row:
        if hasattr(row[k], 'unit'):
            units[k] = row[k].unit
            row[k] = row[k].value

    return row, units


def main(run_name, pool):
    c = Config.from_run_name(run_name)

    # numbers we need to validate
    n_requested = c.requested_samples_per_star
    prior = c.get_prior()
    poly_trend = prior.poly_trend

    for path in [c.joker_results_path, c.tasks_path]:
        if not os.path.exists(path):
            raise IOError(f"File {path} does not exist! Did you run the "
                          "preceding pipeline steps?")

    tasks = []
    with h5py.File(c.tasks_path, 'r') as tasks_f:
        for apogee_id in tasks_f:
            data = tj.RVData.from_timeseries(tasks_f[apogee_id])
            tasks.append((apogee_id, data))

    logger.debug('Loaded {0} tasks...preparing process queue'
                 .format(len(tasks)))

    full_tasks = []
    with h5py.File(c.joker_results_path, 'r') as results_f:
        for apogee_id, data in tasks:
            if apogee_id not in results_f:
                continue

            full_tasks.append([apogee_id, data, poly_trend, n_requested,
                               c.joker_results_path])

    logger.info('Done preparing tasks: {0} stars in process queue'
                .format(len(full_tasks)))

    rows = []
    for r, units in tqdm(pool.starmap(worker, full_tasks),
                         total=len(full_tasks)):
        if r is not None:
            rows.append(r)

    tbl = Table(rows)

    for k, unit in units.items():
        tbl[k] = tbl[k] * unit

    tbl.write(c.metadata_path, overwrite=True)


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='TODO',
                        loggers=logger)

    args = parser.parse_args()

    with args.Pool(**args.Pool_kwargs) as pool:
        main(run_name=args.run_name, pool=pool)

    sys.exit(0)
