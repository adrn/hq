# Standard library
from os import path
import pickle

# Third-party
from astropy.time import Time
import h5py
import matplotlib.pyplot as plt
import numpy as np
from thejoker.sampler import JokerSamples

# Project
from twoface.log import log as logger
from twoface.plot import plot_mcmc_diagnostic, plot_data_orbits
from twoface.samples_analysis import MAP_sample

__all__ = ['gelman_rubin', 'emcee_worker']


def gelman_rubin(chain):
    """
    Implementation from http://joergdietrich.github.io/emcee-convergence.html
    """
    m, n, *_ = chain.shape

    V = np.var(chain, axis=1, ddof=1) # variance over steps
    W = np.mean(V, axis=0) # mean variance over walkers

    θb = np.mean(chain, axis=1) # mean over steps
    θbb = np.mean(θb, axis=0) # mean over walkers

    B = n / (m - 1) * np.sum((θbb - θb)**2, axis=0)
    var_θ = (n - 1) / n * W + 1 / n * B

    return np.sqrt(var_θ / W)


def emcee_worker(task):
    cache_path, results_filename, apogee_id, data, joker = task
    n_walkers = 1024
    n_steps = 32768

    chain_path = path.join(cache_path, '{0}.npy'.format(apogee_id))
    plot_path = path.join(cache_path, '{0}.png'.format(apogee_id))
    orbits_plot_path = path.join(cache_path, '{0}-orbits.png'.format(apogee_id))
    model_path = path.join(cache_path, 'model.pickle')

    sampler = None
    if not path.exists(chain_path):
        logger.debug('Running MCMC for {0}'.format(apogee_id))

        with h5py.File(results_filename, 'r') as f:
            samples0 = JokerSamples.from_hdf5(f[apogee_id])

        sample = MAP_sample(data, samples0, joker.params)
        model, samples, sampler = joker.mcmc_sample(data, sample,
                                                    n_burn=0,
                                                    n_steps=n_steps,
                                                    n_walkers=n_walkers,
                                                    return_sampler=True)

        np.save(chain_path, sampler.chain[:, n_steps//2:].astype('f4'))

        if not path.exists(model_path):
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

    if not path.exists(plot_path):
        logger.debug('Making plots for {0}'.format(apogee_id))

        if sampler is None:
            chain = np.load(chain_path)
        else:
            chain = sampler.chain

        fig = plot_mcmc_diagnostic(chain)
        fig.savefig(plot_path, dpi=250)
        plt.close(fig)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        samples = model.unpack_samples_mcmc(chain[:, -1])
        samples.t0 = Time(data._t0_bmjd, format='mjd', scale='tcb')
        fig = plot_data_orbits(data, samples)
        fig.savefig(orbits_plot_path, dpi=250)
        plt.close(fig)
