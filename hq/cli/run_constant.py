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


def get_robust_constant_model(data):
    with pm.Model() as const_model:
        rv = pm.Data('rv', data['rv'])
        rv_err = pm.Data('rv_err', data['rv_err'])

        mu = pm.Uniform('mu', -500, 500)
        lns = pm.Uniform('lns', -8, 6)
        err = tt.sqrt(rv_err**2 + tt.exp(2 * lns))

        like = pm.Normal('like', mu, err,
                         observed=rv)
        pm.Deterministic('loglike', like.logpt)

    return const_model


def get_robust_linear_model(data):
    with pm.Model() as linear_model:
        t = pm.Data('t', data['t'])
        rv = pm.Data('rv', data['rv'])
        rv_err = pm.Data('rv_err', data['rv_err'])

        a = pm.Uniform('a', -500, 500)
        b = pm.Uniform('b', -500, 500)
        lns = pm.Uniform('lns', -8, 6)
        err = tt.sqrt(rv_err**2 + tt.exp(2 * lns))

        like = pm.Normal('like', a + b*t, err,
                         observed=rv)
        pm.Deterministic('loglike', like.logpt)

    return linear_model


def worker(task):
    source_ids, worker_id, c = task

    logger.debug(f"Worker {worker_id} running {len(source_ids)} stars")

    all_data = {}
    with h5py.File(c.tasks_file, 'r') as tasks_f:
        for source_id in source_ids:
            data = tj.RVData.from_timeseries(tasks_f[source_id])
            all_data[source_id] = data

    rows = []
    for source_id in source_ids:
        data = all_data[source_id]

        dt = (data.t.mjd - np.mean(data.t.mjd)) / 1000.
        rv = data.rv.to_value(u.km/u.s).astype('f8')
        rv_err = data.rv_err.to_value(u.km/u.s).astype('f8')
        data_dict = {'t': dt, 'rv': rv, 'rv_err': rv_err}
        assert all([np.isfinite(x).all() for x in data_dict.values()])
        const_model = get_robust_constant_model(data_dict)
        linear_model = get_robust_linear_model(data_dict)

        init_s = np.sqrt(np.sum((data.rv - np.mean(data.rv))**2 -
                                data.rv_err**2) / len(data))
        init_lns = np.log(init_s.value)
        if not np.isfinite(init_lns):
            init_lns = -6

        init_b = np.polyfit(data_dict['t'], data_dict['rv'], deg=1)[0]
        if np.abs(init_b) > 500:
            init_b = 0.

        # Find the MAP of the robust constant model:
        for i in range(4):  # HACK: try a few iterations...
            try:
                with const_model:
                    const_map_p, const_info = pmx.optimize(
                        start={'mu': np.mean(rv) + np.random.normal(0, 1e-2),
                               'lns': init_lns},
                        method='L-BFGS-B',
                        progress=False, verbose=False, return_info=True)
                if not const_info.success and np.any(np.isnan(const_info.x)):
                    raise RuntimeError("failed to fit")

                break

            except Exception as e:
                logger.error(f"FAILED: robust constant for source {source_id}\n"
                             f"{const_info}\n{e}")
                const_map_p = None

        for i in range(4):  # HACK: try a few iterations...
            try:
                with linear_model:
                    linear_map_p, lin_info = pmx.optimize(
                        start={'a': np.mean(rv),
                               'b': init_b,
                               'lns': init_lns},
                        method='L-BFGS-B',
                        progress=False, verbose=False, return_info=True)
                if not lin_info.success and np.any(np.isnan(lin_info.x)):
                    raise RuntimeError("failed to fit")

                break

            except Exception as e:
                logger.error(f"FAILED: robust linear for source {source_id}\n"
                             f"{lin_info}\n{e}")
                linear_map_p = None

        # Non-robust / standard constant log-likelihood
        eval_mu = np.sum(rv / rv_err**2) / np.sum(1. / rv_err**2)
        constant_ll = norm.logpdf(rv, eval_mu, rv_err).sum()

        resp = dict()
        resp[c.source_id_colname] = source_id

        resp['constant_ln_likelihood'] = constant_ll

        if const_map_p is not None:
            resp['robust_constant_mean'] = const_map_p['mu']
            resp['robust_constant_scatter'] = np.exp(const_map_p['lns'])
            resp['robust_constant_ln_likelihood'] = const_map_p['loglike']
            resp['robust_constant_success'] = True
        else:
            resp['robust_constant_mean'] = np.nan
            resp['robust_constant_scatter'] = np.nan
            resp['robust_constant_ln_likelihood'] = np.nan
            resp['robust_constant_success'] = False

        if linear_map_p is not None:
            resp['robust_linear_a'] = linear_map_p['a']
            resp['robust_linear_b'] = linear_map_p['b']
            resp['robust_linear_scatter'] = np.exp(linear_map_p['lns'])
            resp['robust_linear_ln_likelihood'] = linear_map_p['loglike']
            resp['robust_linear_success'] = True
        else:
            resp['robust_linear_a'] = np.nan
            resp['robust_linear_b'] = np.nan
            resp['robust_linear_scatter'] = np.nan
            resp['robust_linear_ln_likelihood'] = np.nan
            resp['robust_linear_success'] = False

        rows.append(resp)

    logger.debug(f"Worker {worker_id} finished batch")

    return rows


def run_constant(run_path, pool, overwrite=False):
    c = Config(run_path / 'config.yml')

    if c.constant_results_file.exists() and not overwrite:
        logger.info(f"Results file {str(c.constant_results_file)} already "
                    "exists. Use --overwrite if needed")
        return

    if not c.tasks_file.exists():
        raise IOError(f"Tasks file '{str(c.tasks_file)}' does not exist! Did "
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
    tbl.write(c.constant_results_file, overwrite=True)
