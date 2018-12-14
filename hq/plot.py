# Third-party
import astropy.units as u
from astropy.stats import median_absolute_deviation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from thejoker.plot import plot_rv_curves
from thejoker.sampler.likelihood import get_ivar

from .samples_analysis import unimodal_P, max_likelihood_sample

__all__ = ['plot_data_orbits', 'plot_mcmc_diagnostic', 'plot_two_panel']

# TODO: customize this in TheJoker?
_RV_LBL = 'RV [{0:latex_inline}]'


def plot_data_orbits(data, samples, n_orbits=128, jitter=None,
                     xlim_choice='data', ylim_fac=1., n_times=4096, title=None,
                     ax=None, highlight_P_extrema=True, plot_kwargs=None, data_plot_kwargs=None, relative_to_t0=False):
    """
    Plot the APOGEE RV data vs. time and orbits computed from The Joker samples.

    Parameters
    ----------
    data : :class:`~twoface.data.APOGEERVData`
        The radial velocity data.
    samples : :class:`~thejoker.samples.JokerSamples`
        Posterior samples from The Joker.
    n_orbits : int, optional
        Number of orbits to plot over the data.
    jitter : :class:`~astropy.units.Quantity`, optional
        The jitter used to do the sampling. Only relevant if the jitter was
        fixed. Used to inflate the error bars for the data.
    xlim_choice : str, optional
        Multiple options for how to set the x-axis limits for the plot.
        ``xlim_choice = 'wide'`` sets the xlim to be twice the longest period
        sample.
        ``xlim_choice = 'tight'`` sets the xlim to be twice the time span of the
        data.

    """

    if jitter is not None:
        data = data.copy()

        ivar = get_ivar(data, jitter.to(data.rv.unit).value)
        data.ivar = ivar / data.rv.unit**2

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12,6))
    else:
        fig = ax.figure

    w = np.ptp(data.t.mjd)
    if xlim_choice == 'tight': # twice the span of the data
        t_grid = np.linspace(data.t.mjd.min() - w*0.05,
                             data.t.mjd.max() + w*1.05,
                             n_times)

    elif xlim_choice == 'wide': # twice the longest period sample
        t_min = data.t.mjd.min()
        t_max = max(data.t.mjd.min() + 2*samples['P'].max().value,
                    data.t.mjd.max())
        t_grid = np.linspace(t_min - 0.05*w,
                             t_max + 0.05*w,
                             n_times)

    elif xlim_choice == 'data': # span of the data
        t_grid = np.linspace(data.t.mjd.min() - w*0.05,
                             data.t.mjd.max() + w*0.05,
                             n_times)

    else:
        raise ValueError('Invalid xlim_choice {0}. Can be "wide" or "tight".'
                         .format(xlim_choice))

    if plot_kwargs is None:
        plot_kwargs = dict()

    plot_kwargs.setdefault('color', 'tab:blue')
    plot_kwargs.setdefault('linewidth', 0.5)

    if data_plot_kwargs is None:
        data_plot_kwargs = dict()

    data_plot_kwargs.setdefault('zorder', 5)
    data_plot_kwargs.setdefault('elinewidth', 1)

    plot_rv_curves(samples, t_grid, rv_unit=u.km/u.s, data=data, ax=ax,
                   n_plot=min(len(samples['P']), n_orbits),
                   plot_kwargs=plot_kwargs,
                   data_plot_kwargs=data_plot_kwargs,
                   relative_to_t0=relative_to_t0)

    if highlight_P_extrema:
        # Darken the shortest period sample
        dark_style = dict(color='#333333', alpha=0.5, linewidth=1, zorder=10)

        P_min_samples = samples[samples['P'].argmin()]
        plot_rv_curves(P_min_samples, t_grid, rv_unit=u.km/u.s, ax=ax,
                       n_plot=1, plot_kwargs=dark_style,
                       relative_to_t0=relative_to_t0)

        # Darken the longest period sample
        P_max_samples = samples[samples['P'].argmax()]
        plot_rv_curves(P_max_samples, t_grid, rv_unit=u.km/u.s, ax=ax,
                       n_plot=1, plot_kwargs=dark_style,
                       relative_to_t0=relative_to_t0)

    if relative_to_t0:
        ax.set_xlim(t_grid.min() - data.t0.mjd,
                    t_grid.max() - data.t0.mjd)
    else:
        ax.set_xlim(t_grid.min(), t_grid.max())

    _rv = data.rv.to(u.km/u.s).value
    h = np.ptp(_rv)
    ax.set_ylim(_rv.min()-ylim_fac*h, _rv.max()+ylim_fac*h)

    if title is not None:
        ax.set_title(title)

    return fig


def plot_mcmc_diagnostic(chain):
    """
    TODO:
    """

    names = [r'$\ln P$', r'$\sqrt{K}\,\cos M_0$', r'$\sqrt{K}\,\sin M_0$',
             r'$\sqrt{e}\,\cos \omega$', r'$\sqrt{e}\,\sin \omega$',
             r'$\ln s^2$', '$v_0$']

    ndim = chain.shape[-1]
    assert ndim == len(names)

    fig, axes = plt.subplots(ndim, 3, figsize=(12, 16), sharex=True)

    for k in range(ndim):
        axes[k, 0].set_ylabel(names[k])
        axes[k, 0].plot(chain[..., k].T, marker='',
                        drawstyle='steps-mid',
                        alpha=0.1, rasterized=True)
        axes[k, 1].plot(np.median(chain[..., k], axis=0),
                        marker='', drawstyle='steps-mid')

        std = 1.5 * median_absolute_deviation(chain[..., k], axis=0)
        axes[k, 2].plot(std, marker='', drawstyle='steps-mid')

    axes[0, 0].set_title('walkers')
    axes[0, 1].set_title('med(walkers)')
    axes[0, 2].set_title('1.5 MAD(walkers)')

    fig.tight_layout()
    return fig


def plot_two_panel(data, samples, axes=None, tight=True, title=None,
                   plot_data_orbits_kw=None, scatter_kw=None):
    """Make a two-panel plot with (1) the data and orbit samples, and (2) the
    orbit samples in period-eccentricity space.

    Parameters
    ----------
    star : `~twoface.db.AllStar`
    samples : `~thejoker.JokerSamples`
    plot_data_orbits_kw : dict (optional)
    scatter_kw : dict (optional)

    Returns
    -------
    fig : `matplotlib.Figure`

    """
    if plot_data_orbits_kw is None:
        plot_data_orbits_kw = dict()

    if scatter_kw is None:
        scatter_kw = dict()

    if axes is None:
        fig = plt.figure(figsize=(12, 4.3))
        gs = GridSpec(1, 3)
        ax1 = fig.add_subplot(gs[:2])
        ax2 = fig.add_subplot(gs[2])
        axes = [ax1, ax2]

    else:
        ax1, ax2 = axes
        fig = ax1.figure

    # orbits
    plot_data_orbits_kw.setdefault('xlim_choice', 'tight')
    plot_data_orbits_kw.setdefault('highlight_P_extrema', False)
    fig = plot_data_orbits(data, samples, ax=ax1,
                           **plot_data_orbits_kw)
    if title is not None:
        ax1.set_title(title, fontsize=20)

    # orbit samples
    scatter_kw.setdefault('marker', '.')
    scatter_kw.setdefault('alpha', 0.5)
    scatter_kw.setdefault('linewidth', 0.)
    ax2.scatter(samples['P'], samples['e'], **scatter_kw)

    ax2.set_xscale('log', basex=10)
    ax2.set_xlim(0.5, 2**15)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel('period, $P$ [day]')
    ax2.set_ylabel('eccentricity, $e$')
    ax2.xaxis.set_ticks(10**np.arange(0, 4+1, 1))

    if tight:
        fig.tight_layout()

    return fig


def plot_phase_fold(data, sample, ax=None, label=True,
                    jitter_errorbar=True, orbit_style=None):
    """
    TODO:

    Parameters
    ----------
    star : `~twoface.db.AllStar`
    sample : `~thejoker.JokerSamples`

    Returns
    -------
    fig : `matplotlib.Figure`
    """

    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    if len(sample) > 1:
        sample = max_likelihood_sample(data, sample)
        # raise ValueError('Can only pass one sample.')

    if orbit_style is None:
        orbit_style = dict()
    orbit_style.setdefault('color', 'tab:blue')
    orbit_style.setdefault('zorder', -1)

    P = sample['P']
    M0 = sample['M0']
    orbit = sample.get_orbit(0)

    t0 = data.t0 + (P/(2*np.pi)*M0).to(u.day, u.dimensionless_angles())
    phase = data.phase(P=P, t0=t0)

    # plot the phase-folded data and orbit
    rv_unit = u.km/u.s
    ax.errorbar(phase, data.rv.to(rv_unit).value,
                data.stddev.to(rv_unit).value,
                linestyle='none', marker='o', color='k', markersize=5,
                zorder=10)

    if jitter_errorbar:
        ax.errorbar(phase, data.rv.to(rv_unit).value,
                    np.sqrt(data.stddev**2 +
                            sample['jitter']**2).to(rv_unit).value,
                    linestyle='none', marker='', elinewidth=0., color='#aaaaaa',
                    alpha=0.9, capsize=3, capthick=1, zorder=9)

    phase_grid = np.linspace(0, 1, 1024)
    ax.plot(phase_grid, orbit.radial_velocity(t0 + phase_grid*P),
            marker='', **orbit_style)

    if label:
        ax.set_xlabel(r'phase, $\frac{M-M_0}{2\pi}$')
        ax.set_ylabel(_RV_LBL.format(rv_unit))

    return fig


# TODO: add and update function above
def plot_phase_fold_residual(data, sample, axes=None, label=True,
                             jitter_errorbar=True, orbit_style=None):
    """Plot the data phase-folded at the median period of the input samples,
    and the residuals as a function of phase.

    Parameters
    ----------
    star : `~twoface.db.AllStar`
    sample : `~thejoker.JokerSamples`

    Returns
    -------
    fig : `matplotlib.Figure`
    """

    if not unimodal_P(sample, data):
        raise ValueError('multi-modal period distribution')

    if len(sample) > 1:
        raise ValueError('can only pass a single sample to phase-fold at.')

    # HACK: hard-set getting the median
    orbit = sample.get_orbit(0)
    M0 = sample['M0']
    P = sample['P']
    s = sample['jitter']
    t0 = data.t0 + (P/(2*np.pi)*M0).to(u.day, u.dimensionless_angles())
    phase = data.phase(P=P, t0=t0)

    # compute chi^2 of the orbit fit
    residual = data.rv - orbit.radial_velocity(data.t)
    err = np.sqrt(data.stddev**2 + s**2)
    # chisq = np.sum((residual**2 / err**2).decompose())

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    else:
        fig = axes[0].figure

    # plot the phase-folded data and orbit
    rv_unit = u.km/u.s
    axes[0].errorbar(phase, data.rv.to(rv_unit).value,
                     data.stddev.to(rv_unit).value, zorder=10,
                     linestyle='none', marker='o', color='k', markersize=5)

    phase_grid = np.linspace(0, 1, 1024)
    axes[0].plot(phase_grid, orbit.radial_velocity(t0 + phase_grid*P),
                 marker='', zorder=-1, color='#aaaaaa')

    # plot the residuals
    axes[1].errorbar(phase, residual.to(rv_unit).value,
                     data.stddev.to(rv_unit).value,
                     linestyle='none', marker='o', color='k', markersize=5,
                     zorder=10)

    if jitter_errorbar:
        axes[0].errorbar(phase, data.rv.to(rv_unit).value,
                         err.to(rv_unit).value,
                         linestyle='none', marker='', elinewidth=0., color='#aaaaaa', alpha=0.9, capsize=3, capthick=1,
                         zorder=5)
        axes[1].errorbar(phase, residual.to(rv_unit).value,
                         err.to(rv_unit).value,
                         linestyle='none', marker='', elinewidth=0., color='#aaaaaa', alpha=0.9, capsize=3, capthick=1,
                         zorder=5)

    lim = np.abs(axes[1].get_ylim()).max()
    axes[1].set_ylim(-lim, lim)

    if label:
        axes[1].set_xlabel(r'phase, $\frac{M-M_0}{2\pi}$')
        axes[0].set_ylabel(_RV_LBL.format(rv_unit))
        axes[1].set_ylabel('residual [{0:latex_inline}]'.format(rv_unit))

    axes[0].set_xlim(0, 1)
    axes[1].set_xlim(0, 1)

    return fig
