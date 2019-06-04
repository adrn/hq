# Standard library
from os import path
import sys

# Third-party
from astropy.table import Table
import numpy as np
from thejoker.log import log as joker_logger
from tqdm import tqdm
import yaml
from scipy.optimize import minimize
from schwimmbad import SerialPool
from schwimmbad.mpi import MPIAsyncPool

# Project
from hq.data import get_rvdata
from hq.log import logger
from hq.config import HQ_CACHE_PATH, config_to_alldata
from hq.script_helpers import get_parser


def ln_normal(x, mu, var):
    return -0.5 * (np.log(2*np.pi * var) + (x - mu)**2 / var)


def ln_likelihood_robust(p, data):
    mu, lns2, f = p

    rv = data.rv.value.astype('f8')
    var = data.stddev.value.astype('f8') ** 2

    return np.logaddexp(np.log(f) + ln_normal(rv, mu, var),
                        np.log(1-f) + ln_normal(rv, mu, var + np.exp(lns2))).sum()


def neg_ln_likelihood(*args, **kwargs):
    return -ln_likelihood_robust(*args, **kwargs)


def worker(apogee_id, data):
    # First optimize the negative log-likelihood for the robust constant model:
    res = minimize(neg_ln_likelihood,
                   x0=(np.mean(data.rv.value.astype('f8')), -4, 0.75),
                   args=(data, ), method='L-BFGS-B',
                   bounds=[(-500, 500), (-10, 8), (0.5, 1.)],
                   options=dict(ftol=1e-11))
    ll_robust = -res.fun

    opt_mu = np.sum(data.rv.value / data.stddev.value**2) / np.sum(1 / data.stddev.value**2)
    ll_basic = ln_likelihood_robust([opt_mu, -8, 1], data)

    resp = dict()
    resp['apogee_id'] = apogee_id
    resp['robust_constant_ln_likelihood'] = ll_robust
    resp['robust_success'] = res.success
    resp['constant_ln_likelihood'] = ll_basic
    return resp


def main(run_name, pool, overwrite=False, seed=None):
    run_path = path.join(HQ_CACHE_PATH, run_name)
    with open(path.join(run_path, 'config.yml'), 'r') as f:
        config = yaml.load(f.read())

    # Get paths to files needed to run
    # TODO: doesn't handle jitter!
    results_path = path.join(HQ_CACHE_PATH, run_name,
                             'constant-{0}.fits'.format(run_name))

    if path.exists(results_path) and not overwrite:
        logger.info("Results file {} already exists. Use --overwrite if needed"
                    .format(results_path))
        return

    # Get data files out of config file:
    allstar, allvisit = config_to_alldata(config)
    allvisit = allvisit[np.isin(allvisit['APOGEE_ID'], allstar['APOGEE_ID'])]

    tasks = []
    logger.debug("Loading data and preparing tasks...")
    for star in tqdm(allstar):
        visits = allvisit[allvisit['APOGEE_ID'] == star['APOGEE_ID']]
        data = get_rvdata(visits)
        tasks.append([star['APOGEE_ID'], data])

    logger.info('Done preparing tasks: {0} stars in process queue'
                .format(len(tasks)))

    results = []
    for r in tqdm(pool.starmap(worker, tasks), total=len(tasks)):
        results.append(r)

    tbl = Table(results)
    tbl.write(results_path, overwrite=True)


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='Run The Joker on APOGEE data',
                        loggers=[logger, joker_logger])

    parser.add_argument("--name", dest="run_name", required=True,
                        type=str, help="The name of the run.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
                        help="Random number seed")

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true",
                        help="Overwrite any existing samplings")

    args = parser.parse_args()

    if args.mpi:
        Pool = MPIAsyncPool
    else:
        Pool = SerialPool

    with Pool() as pool:
        main(run_name=args.run_name, pool=pool, overwrite=args.overwrite,
             seed=args.seed)

    sys.exit(0)
