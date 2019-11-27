# Standard library
import os
import sys

# Third-party
from astropy.table import Table
import astropy.units as u
import h5py
import numpy as np
from thejoker.logging import logger as joker_logger
from tqdm import tqdm
from scipy.optimize import minimize
from thejoker.data import RVData

# Project
from hq.log import logger
from hq.config import Config
from hq.script_helpers import get_parser


def ln_normal(x, mu, var):
    return -0.5 * (np.log(2*np.pi * var) + (x - mu)**2 / var)


def ln_likelihood_robust(p, rv, var):
    mu, lns2, f = p
    return np.logaddexp(
        np.log(f) + ln_normal(rv, mu, var),
        np.log(1-f) + ln_normal(rv, mu, var + np.exp(lns2))
    ).sum()


def neg_ln_likelihood(*args, **kwargs):
    return -ln_likelihood_robust(*args, **kwargs)


def worker(task):
    apogee_id, data = task

    rv = data.rv.to_value(u.km/u.s).astype('f8')
    var = data.rv_err.to_value(u.km/u.s).astype('f8') ** 2

    # First optimize the negative log-likelihood for the robust constant model:
    res = minimize(neg_ln_likelihood,
                   x0=(np.mean(rv.astype('f8')), -4, 0.75),
                   args=(rv, var), method='L-BFGS-B',
                   bounds=[(-500, 500), (-10, 8), (0.5, 1.)],
                   options=dict(ftol=1e-11))
    ll_robust = -res.fun

    opt_mu = np.sum(rv/var) / np.sum(1/var)
    ll_basic = ln_likelihood_robust([opt_mu, -8, 1], rv, var)

    resp = dict()
    resp['APOGEE_ID'] = apogee_id
    resp['robust_constant_ln_likelihood'] = ll_robust
    resp['robust_success'] = res.success
    resp['constant_ln_likelihood'] = ll_basic
    return resp


def main(run_name, pool, overwrite=False, seed=None):
    c = Config.from_run_name(run_name)

    # Get paths to files needed to run:
    results_path = os.path.join(c.run_path, 'constant.fits')

    if os.path.exists(results_path) and not overwrite:
        logger.info("Results file {} already exists. Use --overwrite if needed"
                    .format(results_path))
        return

    if not os.path.exists(c.tasks_path):
        raise IOError("Tasks file '{0}' does not exist! Did you run "
                      "make_tasks.py?")

    tasks = []
    with h5py.File(c.tasks_path, 'r') as tasks_f:
        for apogee_id in tasks_f:
            data = RVData.from_timeseries(tasks_f[apogee_id])
            tasks.append((apogee_id, data))

    logger.info('Done preparing tasks: {0} stars in process queue'
                .format(len(tasks)))

    results = []
    for r in tqdm(pool.map(worker, tasks), total=len(tasks)):
        results.append(r)

    tbl = Table(results)
    tbl.write(results_path, overwrite=True)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits

    # Define parser object
    parser = get_parser(description='Run The Joker on APOGEE data',
                        loggers=[logger, joker_logger])

    parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
                        help="Random number seed")

    args = parser.parse_args()

    with threadpool_limits(limits=1, user_api='blas'):
        with args.Pool(**args.Pool_kwargs) as pool:
            main(run_name=args.run_name, pool=pool, overwrite=args.overwrite,
                seed=args.seed)

    sys.exit(0)
