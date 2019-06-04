# Third-party
import astropy.units as u
import numpy as np
from scipy.optimize import minimize
from thejoker.sampler.mcmc import TheJokerMCMCModel

__all__ = ['unimodal_P', 'max_likelihood_sample', 'MAP_sample', 'chisq',
           'max_phase_gap', 'phase_coverage', 'periods_spanned',
           'optimize_mode']


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


def chisq(data, samples):
    """Return the chi squared of the samples.

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

    return chisqs


def max_likelihood_sample(data, samples):
    """Return the maximum-likelihood sample.

    Parameters
    ----------
    data : `~thejoker.RVData`
    samples : `~thejoker.JokerSamples`

    """
    return samples[np.argmin(chisq(data, samples))]


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


def max_phase_gap(sample, data):
    """Based on the MPG statistic defined here:
    https://gea.esac.esa.int/archive/documentation/GDR2/Data_analysis/chap_cu7var/sssec_cu7var_validation_sos_sl/ssec_cu7var_sos_sl_qa.html

    Parameters
    ----------
    sample : `~thejoker.JokerSamples`
    data : `~thejoker.RVData`
    """
    phase = np.sort(data.phase(sample['P']))
    return (phase[1:] - phase[:-1]).max()


def phase_coverage(sample, data, n_bins=10):
    """Based on the PC statistic defined here:
    https://gea.esac.esa.int/archive/documentation/GDR2/Data_analysis/chap_cu7var/sssec_cu7var_validation_sos_sl/ssec_cu7var_sos_sl_qa.html

    Parameters
    ----------
    sample : `~thejoker.JokerSamples`
    data : `~thejoker.RVData`
    """
    H, _ = np.histogram(data.phase(sample['P']),
                        bins=np.linspace(0, 1, n_bins+1))
    return (H > 0).sum() / n_bins


def periods_spanned(sample, data):
    """Compute the number of periods spanned by the data

    Parameters
    ----------
    sample : `~thejoker.JokerSamples`
    data : `~thejoker.RVData`
    """
    T = data.t.jd.max() - data.t.jd.min()
    return T / sample['P'].to_value(u.day)


def optimize_mode(init_sample, data, joker, minimize_kwargs=None,
                  return_logprobs=False):
    """Compute the maximum likelihood value within the mode that
    the specified sample is in.

    """
    model = TheJokerMCMCModel(joker.params, data)
    init_p = model.pack_samples(init_sample)

    if minimize_kwargs is None:
        minimize_kwargs = dict()
    res = minimize(lambda *args: -model(args), x0=init_p,
                   method='BFGS', **minimize_kwargs)

    if not res.success:
        return None

    opt_sample = model.unpack_samples(res.x)
    opt_sample.t0 = data.t0

    if return_logprobs:
        pp = model.unpack_samples(res.x, add_units=False)
        ll = model.ln_likelihood(pp).sum()
        lp = model.ln_prior(pp).sum()
        return opt_sample, lp, ll

    else:
        return opt_sample
