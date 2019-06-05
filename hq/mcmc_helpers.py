# Third-party
import numpy as np

__all__ = ['gelman_rubin']


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


def ln_normal(x, mu, var):
    return -0.5 * (np.log(2*np.pi * var) + (x - mu)**2 / var)
