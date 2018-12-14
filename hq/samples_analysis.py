# Third-party
import astropy.units as u
import numpy as np
from thejoker.sampler.mcmc import TheJokerMCMCModel

__all__ = ['unimodal_P', 'max_likelihood_sample', 'MAP_sample']


def unimodal_P(samples, data):
    """Check whether the samples returned are within one period mode.

    Parameters
    ----------
    samples : `~thejoker.JokerSamples`

    Returns
    -------
    is_unimodal : bool
    """

    P_samples = samples['P'].to(u.day).value
    P_min = np.min(P_samples)
    T = np.ptp(data.t.mjd)
    delta = 4*P_min**2 / (2*np.pi*T)

    return np.ptp(P_samples) < delta


def max_likelihood_sample(data, samples):
    """Return the maximum-likelihood sample.

    Parameters
    ----------
    data : `~thejoker.RVData`
    samples : `~thejoker.JokerSamples`

    """
    chisqs = np.zeros(len(samples))

    for i in range(len(samples)):
        orbit = samples.get_orbit(i)
        residual = data.rv - orbit.radial_velocity(data.t)
        err = np.sqrt(data.stddev**2 + samples['jitter'][i]**2)
        chisqs[i] = np.sum((residual**2 / err**2).decompose())

    return samples[np.argmin(chisqs)]


def MAP_sample(data, samples, joker_params, return_index=False):
    """Return the maximum a posteriori sample.

    Parameters
    ----------
    data : `~thejoker.RVData`
    samples : `~thejoker.JokerSamples`
    joker_params : `~thejoker.JokerParams`

    """
    model = TheJokerMCMCModel(joker_params, data)

    ln_ps = np.zeros(len(samples))

    mcmc_p = model.pack_samples_mcmc(samples)
    for i in range(len(samples)):
        ln_ps[i] = model.ln_posterior(mcmc_p[i])

    if return_index:
        idx = np.argmax(ln_ps)
        return samples[idx], idx
    else:
        return samples[np.argmax(ln_ps)]
