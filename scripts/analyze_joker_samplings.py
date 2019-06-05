# Standard library
from collections import defaultdict
from os.path import join, exists
import sys

# Third-party
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
import h5py
import numpy as np
from tqdm import tqdm
import yaml
from thejoker import JokerSamples, TheJoker
from schwimmbad import SerialPool
from schwimmbad.mpi import MPIAsyncPool

# Project
from hq.data import get_rvdata
from hq.log import logger
from hq.config import HQ_CACHE_PATH, config_to_alldata, config_to_jokerparams
from hq.script_helpers import get_parser
from hq.mcmc_helpers import ln_normal
from hq.samples_analysis import (unimodal_P, max_phase_gap, phase_coverage,
                                 periods_spanned, optimize_mode)


def worker(apogee_id, data, joker, poly_trend, n_requested_samples,
           results_path):
    with h5py.File(results_path, 'r') as results_f:
        # Load samples from The Joker and probabilities
        samples = JokerSamples.from_hdf5(results_f[apogee_id],
                                         poly_trend=poly_trend)
        ln_p = results_f[apogee_id]['ln_prior'][:]
        ln_l = results_f[apogee_id]['ln_likelihood'][:]

    row = dict()
    row['APOGEE_ID'] = apogee_id
    row['n_visits'] = len(data)

    MAP_idx = (ln_p + ln_l).argmax()
    MAP_sample = samples[MAP_idx:MAP_idx+1]
    for k in MAP_sample.keys():
        row['MAP_'+k] = MAP_sample[k][0]
    row['t0_bmjd'] = MAP_sample.t0.tcb.mjd

    row['MAP_ln_likelihood'] = ln_l[MAP_idx]
    row['MAP_ln_prior'] = ln_p[MAP_idx]

    if len(samples) == n_requested_samples:
        row['joker_completed'] = True
    else:
        row['joker_completed'] = False

    if unimodal_P(samples, data):
        row['unimodal'] = True
    else:
        row['unimodal'] = False

    row['max_phase_gap'] = max_phase_gap(MAP_sample[0], data)
    row['phase_coverage'] = phase_coverage(MAP_sample[0], data)
    row['periods_spanned'] = periods_spanned(MAP_sample[0], data)

    lls = []
    for i, orbit in enumerate(samples.orbits):
        lls.append(ln_normal(orbit.radial_velocity(data.t).to_value(u.km/u.s),
                             data.rv.to_value(u.km/u.s),
                             data.stddev.to_value(u.km/u.s)**2).sum())
    row['max_unmarginalized_ln_likelihood'] = max(lls)
    
    units = dict()
    for k in row:
        if hasattr(row[k], 'unit'):
            units[k] = row[k].unit
            row[k] = row[k].value

    return row, units


def main(run_name, pool):
    run_path = join(HQ_CACHE_PATH, run_name)
    with open(join(run_path, 'config.yml'), 'r') as f:
        config = yaml.load(f.read())

    # Create an instance of The Joker:
    params = config_to_jokerparams(config)
    joker = TheJoker(params)

    # Get paths to files needed to run
    results_path = join(HQ_CACHE_PATH, run_name,
                        'thejoker-{0}.hdf5'.format(run_name))
    metadata_path = join(HQ_CACHE_PATH, run_name,
                         '{0}-metadata.fits'.format(run_name))

    # Load the data for this run:
    allstar, allvisit = config_to_alldata(config)
    allvisit = allvisit[np.isin(allvisit['APOGEE_ID'], allstar['APOGEE_ID'])]

    n_requested_samples = config['requested_samples_per_star']
    poly_trend = config['hyperparams']['poly_trend']

    if not exists(results_path):
        raise IOError("Results file {0} does not exist! Did you run "
                      "run_apogee.py?".format(results_path))

    tasks = []
    with h5py.File(results_path, 'r') as results_f:
        for apogee_id in results_f.keys():
            # Load data
            visits = allvisit[allvisit['APOGEE_ID'] == apogee_id]
            data = get_rvdata(visits)

            tasks.append([apogee_id, data, joker, poly_trend,
                          n_requested_samples, results_path])

    rows = []
    for r, units in tqdm(pool.starmap(worker, tasks), total=len(tasks)):
        rows.append(r)

    tbl = Table(rows)

    for k, unit in units.items():
        tbl[k] = tbl[k] * unit

    tbl.write(metadata_path, overwrite=True)


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='TODO',
                        loggers=logger)

    parser.add_argument("--name", dest="run_name", required=True,
                        type=str, help="The name of the run.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true",
                        help="Overwrite any existing samplings")

    args = parser.parse_args()

    if args.mpi:
        Pool = MPIAsyncPool
    else:
        Pool = SerialPool

    with Pool() as pool:
        main(run_name=args.run_name, pool=pool)

    sys.exit(0)
