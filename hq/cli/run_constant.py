# Third-party
import theano
theano.config.optimizer = 'None'
theano.config.mode = 'FAST_COMPILE'
theano.config.reoptimize_unpickled_function = False
theano.config.cxx = ""
import pymc3 as pm
import pymc3_ext as pmx
import theano.tensor as tt

from astropy.table import Table
import astropy.units as u
import h5py
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import thejoker as tj
from thejoker.multiproc_helpers import batch_tasks

# Project
from hq.log import logger
from hq.config import Config


def get_robust_constant_model():
    with pm.Model() as model:
        rv = pm.Data('rv', np.zeros(1))
        rv_err = pm.Data('rv_err', np.zeros(1))

        # cluster sizes - prior has some MAGIC NUMBERz
        w = pm.Dirichlet("w", a=np.array([10, 1.]), shape=2)

        mu = pm.Uniform('mu', -500, 500)
        lns = pm.Uniform('lns', -8, 6)
        err2 = tt.sqrt(rv_err**2 + tt.exp(2 * lns))

        mean = tt.stack((mu * tt.ones(rv.shape),
                        mu * tt.ones(rv.shape))).T
        sigma = tt.stack((rv_err, err2)).T

        like = pm.NormalMixture(
            'like', w=w, mu=mean, sigma=sigma,
            observed=rv)

        pm.Deterministic('loglike', like.logpt)

    return model


def worker(task):
    source_ids, worker_id, c = task

    logger.debug(f"Worker {worker_id} running {len(source_ids)} stars")

    model = get_robust_constant_model()

    rows = []
    for source_id in source_ids:
        with h5py.File(c.tasks_path, 'r') as tasks_f:
            data = tj.RVData.from_timeseries(tasks_f[source_id])

        rv = data.rv.to_value(u.km/u.s).astype('f8')
        rv_err = data.rv_err.to_value(u.km/u.s).astype('f8')
        data_dict = {'rv': rv, 'rv_err': rv_err}

        # Find the MAP of the robust constant model:
        try:
            with model:
                pm.set_data(data_dict)
                map_estimate, info = pmx.optimize(
                    start={'mu': np.mean(rv),
                           'lns': -4.,
                           'w': np.array([0.95, 0.05])},
                    method='L-BFGS-B',
                    progress=False, verbose=False, return_info=True)

            # Non-robust / standard constant log-likelihood
            eval_mu = np.sum(rv / rv_err**2) / np.sum(1. / rv_err**2)
            constant_ll = norm.logpdf(rv, eval_mu, rv_err).sum()
        except Exception as e:
            logger.error(f"FAILED for source: {source_id}")
            print(e)
            continue

        resp = dict()
        resp[c.source_id_colname] = source_id
        resp['robust_constant_ln_likelihood'] = map_estimate['loglike']
        resp['robust_success'] = info.success
        resp['constant_ln_likelihood'] = constant_ll
        rows.append(resp)

    logger.debug(f"Worker {worker_id} finished batch")

    return rows


def run_constant(run_path, pool, overwrite=False):
    c = Config(run_path / 'config.yml')

    # Get paths to files needed to run:
    results_path = c.cache_path / 'constant.fits'

    if results_path.exists() and not overwrite:
        logger.info(f"Results file {str(results_path)} already exists. Use "
                    "--overwrite if needed")
        return

    if not c.tasks_path.exists():
        raise IOError(f"Tasks file '{str(c.tasks_path)}' does not exist! Did "
                      "you run hq make_tasks?")

    # Get data files out of config file:
    logger.debug("Loading data...")
    source_ids = np.unique(c.data[c.source_id_colname])

    # Make batches of source IDs:
    logger.debug("Preparing tasks...")
    if len(source_ids) > 10 * pool.size:
        n_tasks = min(16 * pool.size, len(source_ids))
    else:
        n_tasks = pool.size
    tasks = batch_tasks(len(source_ids), n_tasks, arr=source_ids, args=(c, ))
    logger.info(f'Done preparing tasks: split into {len(tasks)} task chunks')

    results = []
    for r in tqdm(pool.map(worker, tasks), total=len(tasks)):
        results.append(r)

    tbl = Table([item for sublist in results for item in sublist])
    tbl.write(results_path, overwrite=True)
