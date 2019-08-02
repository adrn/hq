# Standard library
from os import path
import sys
import glob

# Third-party
import astropy.units as u
from astropy.stats import median_absolute_deviation
from astropy.table import Table
import h5py
import numpy as np
from tqdm import tqdm
import yaml
from scipy.special import logsumexp
from thejoker.sampler.mcmc import TheJokerMCMCModel
from thejoker.sampler.fast_likelihood import batch_marginal_ln_likelihood
from thejoker.sampler.io import pack_prior_samples
from thejoker.likelihood import ln_prior
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


def compute_ll_lp(data, samples, joker_params):
    chunk = np.ascontiguousarray(pack_prior_samples(samples, u.km/u.s)[0])
    lls = np.array(batch_marginal_ln_likelihood(chunk, data, joker_params))
    lps = np.array(ln_prior(samples, joker_params))
    return lls, lps


def worker(apogee_id, data, params, n_samples, chain_file):
    model = TheJokerMCMCModel(params, data)

    samples = np.load(chain_file)
    chain = samples['arr_0']
    lnprob = samples['arr_1']

    R = gelman_rubin(chain)

    row = dict()
    row['APOGEE_ID'] = apogee_id
    row['n_visits'] = len(data)

    row['gelman_rubin_max'] = np.max(R)
    row['gelman_rubin_med'] = np.median(R)

    # HACK: some MAGIC NUMBERs below
    flatlnprob = np.concatenate(lnprob)
    flatchain = np.vstack(chain)

    # Compute the MAP sample for summary vals:
    MAP_idx = flatlnprob.argmax()
    MAP_sample = model.unpack_samples(flatchain[MAP_idx])[0]
    MAP_sample.t0 = data.t0

    # But also compute the maximum likelihood sample and value
    samples = model.unpack_samples(flatchain)
    ll, lp = compute_ll_lp(data, samples, params)
    max_ll_idx = np.array(ll).argmax()
    max_ll_sample = model.unpack_samples(flatchain[max_ll_idx])[0]
    max_ll_sample.t0 = data.t0

    # Compute the evidence, p(D), for the Kepler model using the emcee samples:
    row['kepler_ln_evidence'] = logsumexp(ll + lp) - np.log(len(ll))

    flatchain_thin = np.vstack(chain)
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

    # Use the max marginal likelihood sample
    orbit = max_ll_sample.get_orbit(0)
    var = (data.stddev**2 + max_ll_sample['jitter']**2).to_value((u.km/u.s)**2)
    ll = ln_normal(orbit.radial_velocity(data.t).to_value(u.km/u.s),
                   data.rv.to_value(u.km/u.s),
                   var).sum()
    row['max_unmarginalized_ln_likelihood'] = ll

    units = dict()
    for k in row:
        if hasattr(row[k], 'unit'):
            units[k] = row[k].unit
            row[k] = row[k].value

    # select out a random subset of the requested number of samples:
    idx = np.random.choice(len(thin_samples), size=n_samples, replace=False)
    samples = thin_samples[idx]

    res = dict()
    res['row'] = row
    res['samples'] = samples
    res['apogee_id'] = apogee_id
    res['units'] = units

    return res


def main(run_name, pool):
    run_path = path.join(HQ_CACHE_PATH, run_name)
    with open(path.join(run_path, 'config.yml'), 'r') as f:
        config = yaml.load(f.read())
    n_samples = config['requested_samples_per_star']

    # Create an instance of The Joker:
    params = config_to_jokerparams(config)

    # Get paths to files needed to run
    emcee_metadata_path = path.join(HQ_CACHE_PATH, run_name,
                                    'emcee-metadata.fits')
    emcee_results_path = path.join(HQ_CACHE_PATH, run_name,
                                   'emcee-samples.hdf5')

    chain_filenames = glob.glob(path.join(HQ_CACHE_PATH, run_name,
                                          'emcee', '*.npz'))
    apogee_ids = [path.splitext(path.basename(x))[0] for x in chain_filenames]

    # Load the data for this run:
    allstar, allvisit = config_to_alldata(config)
    allstar = allstar[np.isin(allstar['APOGEE_ID'], apogee_ids)]
    allvisit = allvisit[np.isin(allvisit['APOGEE_ID'], allstar['APOGEE_ID'])]

    tasks = []
    for apogee_id, chain_file in zip(apogee_ids, chain_filenames):
        # Load data
        visits = allvisit[allvisit['APOGEE_ID'] == apogee_id]
        data = get_rvdata(visits)
        tasks.append([apogee_id, data, params, n_samples, chain_file])

    rows = []
    with h5py.File(emcee_results_path, 'a') as results_f:
        for res in tqdm(pool.starmap(worker, tasks), total=len(tasks)):
            rows.append(res['row'])

            # replace existing samples
            if res['apogee_id'] in results_f:
                del results_f[res['apogee_id']]

            g = results_f.create_group(res['apogee_id'])
            res['samples'].to_hdf5(g)

            # TODO: no support for prior / likelihood values...
            # g.create_dataset('ln_prior', data=res['ln_prior'])
            # g.create_dataset('ln_likelihood', data=res['ln_likelihood'])

    tbl = Table(rows)

    for k, unit in res['units'].items():
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
