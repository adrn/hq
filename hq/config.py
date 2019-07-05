import os
from os.path import expanduser, join

# Third-party
from astropy.io import fits
import astropy.units as u
import numpy as np

# Project
from thejoker import JokerParams
from .log import logger

__all__ = ['config_to_jokerparams']

HQ_CACHE_PATH = expanduser(os.environ.get("HQ_CACHE_PATH",
                                          join("~", ".hq")))

if not os.path.exists(HQ_CACHE_PATH):
    os.makedirs(HQ_CACHE_PATH, exist_ok=True)

logger.debug("Using cache path:\n\t {}\nSet the environment variable "
             "'HQ_CACHE_PATH' to change.".format(HQ_CACHE_PATH))


def config_to_jokerparams(config):
    P_min = u.Quantity(*config['hyperparams']['P_min'].split())
    P_max = u.Quantity(*config['hyperparams']['P_max'].split())
    kwargs = dict(P_min=P_min, P_max=P_max)

    if 'poly_trend' in config['hyperparams']:
        kwargs['poly_trend'] = config['hyperparams']['poly_trend']

    if 'jitter' in config['hyperparams']:
        # jitter is fixed to some quantity, specified in config file
        jitter = u.Quantity(*config['hyperparams']['jitter'].split())
        logger.debug('Jitter is fixed to: {0:.2f}'.format(jitter))
        kwargs['jitter'] = jitter

    elif 'jitter_prior_mean' in config['hyperparams']:
        # jitter prior parameters are specified in config file
        jitter_mean = config['hyperparams']['jitter_prior_mean']
        jitter_stddev = config['hyperparams']['jitter_prior_stddev']
        jitter_unit = config['hyperparams']['jitter_prior_unit']
        logger.debug('Sampling in jitter with mean = {0:.2f} (stddev in '
                     'log(var) = {1:.2f}) [{2}]'
                     .format(np.sqrt(np.exp(jitter_mean)),
                             jitter_stddev, jitter_unit))
        kwargs['jitter'] = (jitter_mean, jitter_stddev)
        kwargs['jitter_unit'] = u.Unit(jitter_unit)

    std = config['hyperparams'].get('linear_param_std', None)
    if std is not None:
        Vinv = np.diag(1 / np.array(std)**2)
        kwargs['linear_par_Vinv'] = Vinv

    return JokerParams(**kwargs)


def config_to_prior_cache(config, params):
    if params._fixed_jitter:
        prior_filename = ('P{0:.0f}-{1:.0f}_{2:d}_prior_samples.hdf5'
                          .format(params.P_min.to_value(u.day),
                                  params.P_max.to_value(u.day),
                                  config['prior']['n_samples']))
    else:
        prior_filename = ('P{0:.0f}-{1:.0f}_{2:d}_jitter_prior_samples.hdf5'
                          .format(params.P_min.to_value(u.day),
                                  params.P_max.to_value(u.day),
                                  config['prior']['n_samples']))
    return join(HQ_CACHE_PATH, prior_filename)


def config_to_alldata(config):
    allstar_tbl = fits.getdata(config['data']['allstar'])
    allvisit_tbl = fits.getdata(config['data']['allvisit'])
    logger.log(1, "Opened allstar & allvisit files")

    # Remove bad velocities / NaN / Inf values:
    allvisit_tbl = allvisit_tbl[np.isfinite(allvisit_tbl['VHELIO']) &
                                np.isfinite(allvisit_tbl['VRELERR'])]
    allvisit_tbl = allvisit_tbl[(allvisit_tbl['VRELERR'] < 100.) &
                                (allvisit_tbl['VHELIO'] != -9999)]
    logger.log(1, "Filtered bad/NaN/-9999 data")

    starflag_bits = config['data'].get('starflag_bits', None)
    aspcapflag_bits = config['data'].get('aspcapflag_bits', None)
    min_nvisits = config['data'].get('min_nvisits', 3)
    logger.log(1, "Min. number of visits: {0}".format(min_nvisits))

    if starflag_bits is None:  # use deaults
        # VERY_BRIGHT_NEIGHBOR, SUSPECT_RV_COMBINATION, SUSPECT_BROAD_LINES
        star_starflag_bits = [3, 16, 17]

        # LOW_SNR, PERSIST_HIGH, PERSIST_JUMP_POS, PERSIST_JUMP_NEG
        visit_starflag_bits = star_starflag_bits + [4, 9, 12, 13]

    star_starflag_mask = np.sum(2 ** np.array(star_starflag_bits))
    visit_starflag_mask = np.sum(2 ** np.array(visit_starflag_bits))
    logger.log(1, "Using STARFLAG bitmask for allStar: {0}"
               .format(star_starflag_mask))
    logger.log(1, "Using STARFLAG bitmask for allVisit: {0}"
               .format(visit_starflag_mask))
    allvisit_tbl = allvisit_tbl[(allvisit_tbl['STARFLAG'] & visit_starflag_mask) == 0]

    # After quality and bitmask cut, figure out what APOGEE_IDs remain
    v_apogee_ids, counts = np.unique(allvisit_tbl['APOGEE_ID'],
                                     return_counts=True)
    star_mask = np.isin(allstar_tbl['APOGEE_ID'],
                        v_apogee_ids[counts >= min_nvisits])

    if aspcapflag_bits is None:  # use defaults
        # TEFF_BAD, LOGG_BAD, VMICRO_BAD, ROTATION_BAD, VSINI_BAD
        aspcapflag_bits = [16, 17, 18, 26, 30]

    aspcapflag_mask = np.sum(2 ** np.array(aspcapflag_bits))
    logger.log(1, "Using ASPCAPFLAG bitmask: {0}".format(aspcapflag_mask))
    star_mask &= ((allstar_tbl['ASPCAPFLAG'] & aspcapflag_mask) == 0)
    star_mask &= ((allstar_tbl['STARFLAG'] & star_starflag_mask) == 0)
    allstar_tbl = allstar_tbl[star_mask]

    # Only load visits for stars that we're loading
    allvisit_tbl = allvisit_tbl[np.isin(allvisit_tbl['APOGEE_ID'],
                                        allstar_tbl['APOGEE_ID'])]
    v_apogee_ids2 = np.unique(allvisit_tbl['APOGEE_ID'])
    star_mask2 = np.isin(allstar_tbl['APOGEE_ID'], v_apogee_ids2)
    allstar_tbl = allstar_tbl[star_mask2]

    _, idx = np.unique(allstar_tbl['APOGEE_ID'], return_index=True)
    allstar_tbl = allstar_tbl[idx]

    return allstar_tbl, allvisit_tbl
