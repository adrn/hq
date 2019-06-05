# Standard library
from os import path
import sys
import pickle
import glob

# Third-party
import astropy.units as u
from astropy.stats import median_absolute_deviation
from astropy.table import Table
import numpy as np
from tqdm import tqdm
import yaml
from thejoker.sampler.mcmc import TheJokerMCMCModel
from schwimmbad import SerialPool
from schwimmbad.mpi import MPIAsyncPool

# Project
from hq.data import get_rvdata
from hq.log import logger
from hq.config import HQ_CACHE_PATH, config_to_alldata, config_to_jokerparams
from hq.script_helpers import get_parser
from hq.mcmc_helpers import ln_normal, gelman_rubin
from hq.samples_analysis import (max_phase_gap, phase_coverage,
                                 periods_spanned, phase_coverage_per_period)


def worker(apogee_id, data, params, sampler_file):
    model = TheJokerMCMCModel(params, data)

    with open(sampler_file, 'rb') as f:
        emcee_sampler = pickle.load(f)

    chain = emcee_sampler.chain[:, 1024:]
    R = gelman_rubin(chain)

    row = dict()
    row['APOGEE_ID'] = apogee_id
    row['n_visits'] = len(data)

    row['R_max'] = np.max(R)
    row['R_med'] = np.median(R)

    # HACK: some MAGIC NUMBERs below
    flatlnprob = np.concatenate(emcee_sampler.lnprobability[1024:])
    flatchain = np.vstack(emcee_sampler.chain[:, 1024:])
    MAP_idx = flatlnprob.argmax()
    MAP_sample = model.unpack_samples(flatchain[MAP_idx])[0]

    flatchain_thin = np.vstack(emcee_sampler.chain[:, 1024::8])
    thin_samples = model.unpack_samples(flatchain_thin)
    thin_samples.t0 = data.t0

    err = dict()
    for k in thin_samples.keys():
        err[k] = 1.5 * median_absolute_deviation(thin_samples[k])

    for k in MAP_sample.keys():
        row['MAP_{}'.format(k)] = MAP_sample[k]
        row['MAP_{}_err'.format(k)] = err[k]

    row['t0_bmjd'] = data.t0.tcb.mjd

    row['max_phase_gap'] = max_phase_gap(MAP_sample, data)
    row['phase_coverage'] = phase_coverage(MAP_sample, data)
    row['periods_spanned'] = periods_spanned(MAP_sample, data)
    row['phase_coverage_per_period'] = phase_coverage_per_period(MAP_sample,
                                                                 data)

    orbit = MAP_sample.get_orbit(0)
    ll = ln_normal(orbit.radial_velocity(data.t).to_value(u.km/u.s),
                   data.rv.to_value(u.km/u.s),
                   data.stddev.to_value(u.km/u.s)**2).sum()
    row['max_unmarginalized_ln_likelihood'] = ll

    units = dict()
    for k in row:
        if hasattr(row[k], 'unit'):
            units[k] = row[k].unit
            row[k] = row[k].value

            if '{}_err'.format(k) in row.keys():
                row['{}_err'.format(k)] = row['{}_err'.format(k)].to_value(units[k])

    return row, units


def main(run_name, pool):
    run_path = path.join(HQ_CACHE_PATH, run_name)
    with open(path.join(run_path, 'config.yml'), 'r') as f:
        config = yaml.load(f.read())

    # Create an instance of The Joker:
    params = config_to_jokerparams(config)

    # Get paths to files needed to run
    emcee_metadata_path = path.join(HQ_CACHE_PATH, run_name,
                                    '{0}-emcee-metadata.fits'.format(run_name))

    sampler_filenames = glob.glob(path.join(HQ_CACHE_PATH, run_name,
                                            'emcee', '2M*.pickle'))
    apogee_ids = [path.splitext(path.basename(x))[0] for x in sampler_filenames]

    # Load the data for this run:
    allstar, allvisit = config_to_alldata(config)
    allstar = allstar[np.isin(allstar['APOGEE_ID'], apogee_ids)]
    allvisit = allvisit[np.isin(allvisit['APOGEE_ID'], allstar['APOGEE_ID'])]

    tasks = []
    for apogee_id, sampler_file in zip(apogee_ids, sampler_filenames):
        # Load data
        visits = allvisit[allvisit['APOGEE_ID'] == apogee_id]
        data = get_rvdata(visits)
        tasks.append([apogee_id, data, params, sampler_file])

    rows = []
    for r, units in tqdm(pool.starmap(worker, tasks), total=len(tasks)):
        rows.append(r)

    tbl = Table(rows)

    for k, unit in units.items():
        tbl[k] = tbl[k] * unit

    tbl.write(emcee_metadata_path, overwrite=True)


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
