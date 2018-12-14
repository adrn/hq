import os
from os import path
import sys
import time

# Third-party
from astropy.io import fits
from astropy.table import Table, join
from astropy.time import Time
import astropy.units as u
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import schwimmbad

from thejoker.data import RVData
from thejoker.sampler import JokerParams, TheJoker, JokerSamples
from thejoker.plot import plot_rv_curves
from thejoker.log import log as logger
from thejoker.utils import quantity_to_hdf5

from twoface.sample_prior import make_prior_cache

# Default root data path containing APOGEE_DR13/14 paths
APOGEE_DATA_PATH = path.expanduser('~/data/')
# CACHE_PATH = path.abspath(path.join(path.dirname(path.abspath(__file__)),
#                           '..', '..', 'cache'))
CACHE_PATH = path.join(path.dirname(path.abspath(__file__)), 'cache')
PLOT_PATH = path.join(path.dirname(path.abspath(__file__)), 'plots')
for P in [CACHE_PATH, PLOT_PATH]:
    os.makedirs(P, exist_ok=True)

# Columns to read in from the allStar and allVisit files:
star_columns = ['APOGEE_ID', 'NVISITS',
                'TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR', 'M_H', 'M_H_ERR']
visit_columns = ['VISIT_ID', 'APOGEE_ID', 'MJD', 'JD', 'VREL', 'VRELERR',
                 'VHELIO', 'SNR', 'CHISQ']

def read_table(filename, columns):
    """
    Read the specified columns from the specified FITS file and return an
    Astropy Table object.
    """
    tbl = fits.getdata(filename)
    return Table(tbl.view(tbl.dtype, np.ndarray)[columns])

def load_apogee_dr13_dr14_rg(data_path):
    if data_path is None:
        data_path = APOGEE_DATA_PATH

    else:
        data_path = path.expanduser(data_path)

    allstar_dr13 = read_table(path.join(data_path, 'APOGEE_DR13',
                                        'allStar-l30e.2.fits'),
                              star_columns)
    allvisit_dr13 = read_table(path.join(data_path, 'APOGEE_DR13',
                                         'allVisit-l30e.2.fits'),
                               visit_columns)

    allstar_dr14 = read_table(path.join(data_path, 'APOGEE_DR14',
                                        'allStar-l31c.2.fits'),
                              star_columns)
    allvisit_dr14 = read_table(path.join(data_path, 'APOGEE_DR14',
                                         'allVisit-l31c.2.fits'),
                               visit_columns)

    _, uniq_idx = np.unique(allstar_dr13['APOGEE_ID'], return_index=True)
    dr13 = join(allvisit_dr13, allstar_dr13[uniq_idx], join_type='left',
                keys='APOGEE_ID')

    _, uniq_idx = np.unique(allstar_dr14['APOGEE_ID'], return_index=True)
    dr14 = join(allvisit_dr14, allstar_dr14[uniq_idx], join_type='left',
                keys='APOGEE_ID')

    both = join(dr13, dr14,
                join_type="inner", keys=['APOGEE_ID', 'JD'],
                table_names=['dr13', 'dr14'])
    assert np.all(both['MJD_dr13'] == both['MJD_dr14'])

    # now restrict to good visits, and Red Giant stars
    mask = ((both['LOGG_dr14'] < 3) & (both['LOGG_dr14'] > -999) &
            np.isfinite(both['VHELIO_dr13']) & np.isfinite(both['VHELIO_dr14']))
    logger.debug("{0} good visits in both DR13 and DR14 for red giants"
                 .format(mask.sum()))

    return both[mask]

def rvdata_from_rows(rows, name):
    return RVData(
        t=Time(rows['JD'], format='jd', scale='utc'),
        rv=np.array(rows['VHELIO_{0}'.format(name)]).astype('<f8') * u.km/u.s,
        stddev=np.array(rows['VRELERR_{0}'.format(name)]).astype('<f8') * u.km/u.s)

def main(data_path, pool, overwrite=False):

    # Configuration settings / setup
    n_requested_samples = 128
    n_visits = [4, 8, 16]
    n_per_bin = 16 # 16 stars per n_visits bin
    prior_file = path.join(CACHE_PATH, 'dr13_dr14_prior_samples.h5')
    params = JokerParams(P_min=8*u.day, P_max=32768*u.day,
                         jitter=(10., 4.), jitter_unit=u.m/u.s)

    # Load the red giant visits that overlap dr13-dr14
    logger.debug("Loading APOGEE DR13 and DR14 data")
    dr1314 = load_apogee_dr13_dr14_rg(data_path=data_path)

    apogee_ids_file = path.join(CACHE_PATH, 'apogee_ids.npy')
    if path.exists(apogee_ids_file):
        logger.debug("APOGEE ID file already exists")
        arr = np.load(apogee_ids_file)

    else:
        logger.info("Generating APOGEE ID file - this could take a few minutes")

        # Get a pandas dataframe so we can efficiently group visits by APOGEE_ID
        df = dr1314.to_pandas()
        grouped = df.groupby('APOGEE_ID')

        rows = []
        for n in n_visits:
            visits = grouped.filter(lambda x: len(x) == n)
            aids = np.array(visits['APOGEE_ID']).astype(str)
            _ids = np.random.choice(aids, replace=False, size=n_per_bin)
            rows += [(n, _id) for _id in _ids]
        arr = np.array(rows, dtype=[('n_visits', int), ('APOGEE_ID', 'S32')])
        np.save(apogee_ids_file, arr)

    # Make a file to store samples
    dr13_results_filename = path.join(CACHE_PATH, "dr13_samples.hdf5")
    dr14_results_filename = path.join(CACHE_PATH, "dr14_samples.hdf5")

    # Ensure that the results file exists - this is where we cache samples that
    # pass the rejection sampling step
    for results_filename in [dr13_results_filename, dr14_results_filename]:
        if not os.path.exists(results_filename):
            with h5py.File(results_filename, 'w') as f:
                pass

    for row in arr:
        apogee_id = row['APOGEE_ID'].astype(str)
        logger.debug("Running star {0} with {1} visits"
                     .format(apogee_id, row['n_visits']))
        visits = dr1314[dr1314['APOGEE_ID'] == apogee_id]

        data_dr13 = rvdata_from_rows(visits, 'dr13')
        data_dr14 = rvdata_from_rows(visits, 'dr14')

        # plot data
        fig,ax = plt.subplots(1, 1, figsize=(8,6))
        data_dr13.plot(ax=ax, color='tab:blue')
        data_dr14.plot(ax=ax, color='tab:orange')
        fig.savefig(path.join(PLOT_PATH, '{0}_{1}_data.png'
                              .format(row['n_visits'], apogee_id)),
                    dpi=256)

        # Check whether these samples are already in the results file
        done = []
        for results_filename in [dr13_results_filename, dr14_results_filename]:
            with h5py.File(results_filename, 'r+') as f:
                if apogee_id in f or overwrite:
                    done.append(True)

                else:
                    done.append(False)
        both_done = all(done)

        # Only run the joker if we don't already have samples
        if both_done:
            logger.info("Samples already done for DR13 and DR14 - loading...")
            with h5py.File(dr13_results_filename, 'r+') as f:
                samples_dr13 = JokerSamples.from_hdf5(
                    f[apogee_id], trend_cls=params.trend_cls)

            with h5py.File(dr14_results_filename, 'r+') as f:
                samples_dr14 = JokerSamples.from_hdf5(
                    f[apogee_id], trend_cls=params.trend_cls)

        else:
            # Make the prior cache if it doesn't already exist
            if not path.exists(prior_file):
                logger.info("Generating prior cache - this could take a few minutes")
                joker = TheJoker(params)
                make_prior_cache(prior_file, joker,
                                 nsamples=2**29, batch_size=2**24)

            joker = TheJoker(params, pool=pool)

            logger.debug("Beginning sampling...")

            t1 = time.time()
            samples_dr13 = joker.iterative_rejection_sample(
                data_dr13, n_requested_samples=n_requested_samples,
                prior_cache_file=prior_file)
            logger.debug("Done with DR13...{0:.2f} seconds"
                         .format(time.time()-t1))

            t1 = time.time()
            samples_dr14 = joker.iterative_rejection_sample(
                data_dr14, n_requested_samples=n_requested_samples,
                prior_cache_file=prior_file)
            logger.debug("Done with DR14...{0:.2f} seconds"
                         .format(time.time()-t1))

            logger.info("{0} DR13 samples, {1} DR14 samples"
                        .format(len(samples_dr13), len(samples_dr14)))

            for results_filename, samples_dict in zip([dr13_results_filename,
                                                       dr14_results_filename],
                                                      [samples_dr13,
                                                       samples_dr14]):
                # Save the samples to the results file:
                with h5py.File(results_filename, 'r+') as f:
                    if apogee_id not in f:
                        g = f.create_group(apogee_id)

                    else:
                        g = f[apogee_id]

                    for key in samples_dict.keys():
                        if key in g:
                            del g[key]
                        quantity_to_hdf5(g, key, samples_dict[key])

        # plot sample orbits
        n_plot = 128

        span = np.ptp(data_dr13.t.mjd)
        t_grid = np.linspace(data_dr13.t.mjd.min()-0.5*span,
                             data_dr13.t.mjd.max()+0.5*span,
                             1024)

        fig, axes = plt.subplots(2, 1, figsize=(8,10), sharex=True, sharey=True)
        axes[0].set_xlim(t_grid.min(), t_grid.max())

        _ = plot_rv_curves(samples_dr13, t_grid, rv_unit=u.km/u.s,
                           data=data_dr13,
                           ax=axes[0], add_labels=False,
                           n_plot=min([n_plot, len(samples_dr13)]),
                           plot_kwargs=dict(color='#888888', rasterized=True))

        _ = plot_rv_curves(samples_dr14, t_grid, rv_unit=u.km/u.s,
                           data=data_dr14, ax=axes[1],
                           n_plot=min([n_plot, len(samples_dr14)]),
                           plot_kwargs=dict(color='#888888', rasterized=True))

        rv_min = min(data_dr13.rv.to(u.km/u.s).value.min(),
                     data_dr14.rv.to(u.km/u.s).value.min())
        rv_max = max(data_dr13.rv.to(u.km/u.s).value.max(),
                     data_dr14.rv.to(u.km/u.s).value.max())
        yspan = rv_max-rv_min

        axes[0].set_ylim(rv_min-yspan, rv_max+yspan)

        axes[0].set_title('DR13')
        axes[1].set_title('DR14')

        fig.set_facecolor('w')
        fig.tight_layout()
        fig.savefig(path.join(PLOT_PATH, '{0}_{1}_orbits.png'
                              .format(row['n_visits'], apogee_id)),
                    dpi=128)

        # plot samples themselves
        prior_logs2_samples = np.random.normal(params.jitter[0],
                                               params.jitter[1],
                                               size=100000)
        prior_jitter_samples = np.sqrt(np.exp(prior_logs2_samples)) / 1000. # km/s

        fig, axes = plt.subplots(2, 2, figsize=(10, 8),
                                 sharex='col', sharey='col')

        axes[0,0].scatter(samples_dr13['P'].value,
                          samples_dr13['K'].to(u.km/u.s).value,
                          marker='.', color='k', alpha=0.45)

        axes[1,0].scatter(samples_dr14['P'].value,
                          samples_dr14['K'].to(u.km/u.s).value,
                          marker='.', color='k', alpha=0.45)

        axes[1,0].set_xlabel("$P$ [day]")
        axes[0,0].set_ylabel("$K$ [{0:latex_inline}]".format(u.km/u.s))
        axes[1,0].set_ylabel("$K$ [{0:latex_inline}]".format(u.km/u.s))
        axes[0,0].set_xscale('log')
        axes[0,0].set_yscale('log')
        axes[0,0].set_ylim(samples_dr13['K'].to(u.km/u.s).value.min(),
                           samples_dr13['K'].to(u.km/u.s).value.max())

        # jitter

        bins = np.logspace(-5, 1, 32)

        axes[0,1].hist(prior_jitter_samples, bins=bins,
                       normed=True, zorder=-100, color='#666666')
        axes[1,1].hist(prior_jitter_samples, bins=bins,
                       normed=True, zorder=-100, color='#666666')

        axes[0,1].hist(samples_dr13['jitter'].to(u.km/u.s).value, bins=bins,
                       normed=True, alpha=0.6)
        axes[1,1].hist(samples_dr14['jitter'].to(u.km/u.s).value, bins=bins,
                       normed=True, alpha=0.6)
        axes[0,1].set_xscale('log')
        axes[1,1].set_xlabel('jitter, $s$ [{0:latex_inline}]'.format(u.km/u.s))
        axes[0,1].set_xlim(1E-5, 1)

        fig.tight_layout()
        fig.savefig(path.join(PLOT_PATH, '{0}_{1}_samples.png'
                              .format(row['n_visits'], apogee_id)),
                    dpi=256)

        plt.close('all')

    pool.close()
    sys.exit(0)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0,
                          dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0,
                          dest='quietness')

    # multiprocessing
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    # Note: default seed is set!
    parser.add_argument('-s', '--seed', dest='seed', default=42,
                        type=int, help='Random number generator seed.')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        dest='overwrite', default=False,
                        help='Destroy everything.')

    parser.add_argument('--data-path', dest='data_path', default=None, type=str)

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbosity != 0:
        if args.verbosity == 1:
            logger.setLevel(logging.DEBUG)
        else: # anything >= 2
            logger.setLevel(1)

    elif args.quietness != 0:
        if args.quietness == 1:
            logger.setLevel(logging.WARNING)
        else: # anything >= 2
            logger.setLevel(logging.ERROR)

    else: # default
        logger.setLevel(logging.INFO)

    if args.seed is not None:
        np.random.seed(args.seed)

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    main(data_path=args.data_path, pool=pool, overwrite=args.overwrite)
