# Third-party
from astropy.time import Time
import astropy.units as u
import numpy as np

__all__ = ['unimodal_P', 'max_likelihood_sample', 'MAP_sample',
           'max_phase_gap', 'phase_coverage', 'periods_spanned',
           'phase_coverage_per_period']


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
    P = sample['P']
    dt = (data.t - data.t0).to(u.day)
    phase = dt / P
    H1, _ = np.histogram(phase, bins=np.arange(0, phase.max()+1, 1))
    H2, _ = np.histogram(phase, bins=np.arange(-0.5, phase.max()+1, 1))
    return max(H1.max(), H2.max())


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
    sample = JokerSamples(t0=Time(row['t0_bmjd'], format='mjd', scale='tcb'))

    for colname in row.colnames:
        if not colname.startswith('MAP_'):
            continue
        if colname.endswith('_err'):
            continue

        sample[colname[4:]] = u.Quantity([row[colname]])

    return sample[0]


def is_n_modal(data, samples, n_clusters=2):
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=n_clusters)

    lnP = np.log(samples['P'].value).reshape(-1, 1)
    y = clf.fit_predict(lnP)

    unimodals = []
    means = []
    for j in np.unique(y):
        sub_samples = samples[y == j]
        if len(sub_samples) == 1:
            unimodals.append(True)
            means.append(sub_samples)
        else:
            unimodals.append(unimodal_P(sub_samples, data))
            means.append(MAP_sample(sub_samples))

    return all(unimodals), means
