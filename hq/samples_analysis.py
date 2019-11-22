# Third-party
from astropy.time import Time
import astropy.units as u
import numpy as np
from scipy.optimize import minimize
from thejoker.sampler.mcmc import TheJokerMCMCModel
from thejoker import JokerSamples

__all__ = ['unimodal_P', 'max_likelihood_sample', 'MAP_sample', 'chisq',
           'max_phase_gap', 'phase_coverage', 'periods_spanned',
           'phase_coverage_per_period', 'optimize_mode']


def ln_normal(x, mu, var):
    return -0.5 * (np.log(2*np.pi * var) + (x - mu)**2 / var)


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


def ln_likelihood(data, samples):
    """
    Compute the un-marginalized likelihood.

    Parameters
    ----------
    data : `~thejoker.RVData`
    samples : `~thejoker.JokerSamples`

    """
    lls = np.zeros(len(samples))

    for i in range(len(samples)):
        orbit = samples.get_orbit(i)
        var = data.stddev**2 + samples['s'][i]**2
        lls[i] = ln_likelihood(data.rv,
                               orbit.radial_velocity(data.t),
                               var).decompose()

    return lls


def max_likelihood_sample(data, samples):
    """Return the maximum-likelihood sample.

    Parameters
    ----------
    data : `~thejoker.RVData`
    samples : `~thejoker.JokerSamples`

    """
    return samples[np.argmax(ln_likelihood(data, samples))]


def MAP_sample(samples, return_index=False):
    """Return the maximum a posteriori sample.

    Parameters
    ----------
    samples : `~thejoker.JokerSamples`

    """
    if ('ln_prior' not in samples.tbl.colnames
            or 'ln_likelihood' not in samples.tbl.colnames):
        raise ValueError("You must pass in samples that have prior and "
                         "likelihood information stored; use return_logprobs="
                         "True when generating the samples.")

    ln_post = samples['ln_prior'] + samples['ln_likelihood']
    idx = np.argmax(ln_post)

    if return_index:
        return samples[idx], idx
    else:
        return samples[idx]


def max_phase_gap(sample, data):
    """Based on the MPG statistic defined here:
    https://gea.esac.esa.int/archive/documentation/GDR2/Data_analysis/chap_cu7var/sssec_cu7var_validation_sos_sl/ssec_cu7var_sos_sl_qa.html

    Parameters
    ----------
    sample : `~thejoker.JokerSamples`
    data : `~thejoker.RVData`
    """
    phase = np.sort(data.phase(sample['P']))
    phase = np.concatenate((phase, phase))
    return (phase[1:] - phase[:-1]).max()


def phase_coverage(sample, data, n_bins=10):
    """Based on the PC statistic defined here:
    https://gea.esac.esa.int/archive/documentation/GDR2/Data_analysis/chap_cu7var/sssec_cu7var_validation_sos_sl/ssec_cu7var_sos_sl_qa.html

    Parameters
    ----------
    sample : `~thejoker.JokerSamples`
    data : `~thejoker.RVData`
    """
    if len(sample) == 1:
        P = sample['P'][0]
    else:
        P = sample['P']

    H, _ = np.histogram(data.phase(P),
                        bins=np.linspace(0, 1, n_bins+1))
    return (H > 0).sum() / n_bins


def periods_spanned(sample, data):
    """Compute the number of periods spanned by the data

    Parameters
    ----------
    sample : `~thejoker.JokerSamples`
    data : `~thejoker.RVData`
    """
    if len(sample) == 1:
        P = sample['P'][0]
    else:
        P = sample['P']

    T = data.t.jd.max() - data.t.jd.min()
    return T / P.to_value(u.day)


def phase_coverage_per_period(sample, data):
    """The maximum number of data points within a period.

    Parameters
    ----------
    sample : `~thejoker.JokerSamples`
    data : `~thejoker.RVData`
    """
    if len(sample) == 1:
        P = sample['P'][0]
    else:
        P = sample['P']

    dt = (data.t - data.t0).to(u.day)
    phase = dt / P
    H1, _ = np.histogram(phase, bins=np.arange(0, phase.max()+1, 1))
    H2, _ = np.histogram(phase, bins=np.arange(-0.5, phase.max()+1, 1))
    return max(H1.max(), H2.max())


def optimize_mode(init_sample, data, joker, minimize_kwargs=None,
                  return_logprobs=False):
    """Compute the maximum likelihood value within the mode that
    the specified sample is in.

    TODO: rewrite this

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


def constant_model_evidence(data):
    """
    Compute the Bayesian evidence, p(D), for a model that assumes a constant RV
    for the input data.
    """
    N = len(data)
    sn = data.stddev.value
    vn = data.rv.value

    sig2 = 1 / np.sum(1 / sn**2)
    mu = sig2 * np.sum(vn / sn**2)

    Z2 = -0.5 * (N*np.log(2*np.pi) + 2*np.sum(np.log(sn)) -
                 np.log(sig2) + np.sum(vn**2 / sn**2) - mu**2 / sig2)

    return Z2


def extract_MAP_orbit(row):
    data = dict()
    for colname in row.colnames:
        if not colname.startswith('MAP_'):
            continue
        if colname.endswith('_err'):
            continue
        if colname[4:].startswith('ln_'):
            continue

        data[colname[4:]] = row[colname]

    sample = JokerSamples(t0=Time(row['t0_bmjd'], format='mjd', scale='tcb'),
                          **data)
    return sample.get_orbit(0)
