# Third-party
from astropy.stats import median_absolute_deviation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from thejoker.plot import plot_rv_curves

__all__ = ['plot_mcmc_diagnostic', 'plot_two_panel']

# TODO: customize this in TheJoker?
_RV_LBL = 'RV [{0:latex_inline}]'


def plot_mcmc_diagnostic(chain):
    """
    TODO:
    """

    names = [r'$\ln P$', r'$\sqrt{K}\,\cos M_0$', r'$\sqrt{K}\,\sin M_0$',
             r'$\sqrt{e}\,\cos \omega$', r'$\sqrt{e}\,\sin \omega$',
             '$v_0$']

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
                   plot_rv_curves_kw=None, scatter_kw=None):
    """Make a two-panel plot with (1) the data and orbit samples, and (2) the
    orbit samples in period-eccentricity space.

    Parameters
    ----------
    star : `~twoface.db.AllStar`
    samples : `~thejoker.JokerSamples`
    plot_rv_curves_kw : dict (optional)
    scatter_kw : dict (optional)

    Returns
    -------
    fig : `matplotlib.Figure`

    """
    if plot_rv_curves_kw is None:
        plot_rv_curves_kw = dict()

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
    fig = plot_rv_curves(samples, data=data, ax=ax1,
                         **plot_rv_curves_kw)
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
