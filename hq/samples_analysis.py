# Third-party
from astropy.time import Time
import astropy.units as u
import numpy as np


__all__ = ['max_likelihood_sample', 'MAP_sample',
           'max_phase_gap', 'phase_coverage', 'periods_spanned',
           'phase_coverage_per_period']


def ln_normal(x, mu, var):
    return -0.5 * (np.log(2*np.pi * var) + (x - mu)**2 / var)


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


def constant_model_evidence(data):
    """
    Compute the Bayesian evidence, p(D), for a model that assumes a constant RV
    for the input data.
    """
    N = len(data)
    vn = data.rv.value
    sn = data.rv_err.to_value(data.rv.unit)

    sig2 = 1 / np.sum(1 / sn**2)
    mu = sig2 * np.sum(vn / sn**2)

    Z2 = -0.5 * (N*np.log(2*np.pi) + 2*np.sum(np.log(sn)) -
                 np.log(sig2) + np.sum(vn**2 / sn**2) - mu**2 / sig2)

    return Z2


def extract_MAP_sample(row):
    from thejoker import JokerSamples
    sample = JokerSamples(t_ref=Time(row['t_ref_bmjd'],
                                     format='mjd',
                                     scale='tcb'))

    for colname in row.colnames:
        if not colname.startswith('MAP_'):
            continue
        if colname.endswith('_err'):
            continue
        if colname[4:] not in sample._valid_units:
            continue

        sample[colname[4:]] = u.Quantity([row[colname]])

    return sample[0]
