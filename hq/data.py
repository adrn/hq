# Third-party
from astropy.time import Time
import astropy.units as u
import numpy as np
from thejoker import RVData
from .log import logger


def get_new_err(visits, a, b, s):
    err = visits['VRELERR']
    snr = visits['SNR']
    return np.sqrt(s**2 + err**2 + a*snr**b)


def get_rvdata(visits, apply_error_calibration=True, float64=True,
               assume_tcb=False):
    if float64:
        dtype = 'f8'
    else:
        dtype = 'f4'

    t = Time(visits['JD'].astype(dtype), format='jd', scale='utc')
    if assume_tcb:
        t = t.mjd
    rv = visits['VHELIO'].astype(dtype) * u.km/u.s

    if apply_error_calibration:
        # see notebook: Calibrate-visit-rv-err.ipynb
        rv_err = get_new_err(visits,
                             a=145.869, b=-2.8215, s=0.168)
    else:
        rv_err = visits['VRELERR']

    return RVData(t=t, rv=rv, rv_err=rv_err.astype(dtype) * u.km/u.s)


def filter_alldata(allstar_tbl, allvisit_tbl,
                   starflag_bits=None, aspcapflag_bits=None,
                   min_nvisits=3):
    logger.log(1, "Opened allstar & allvisit files")

    # Remove bad velocities / NaN / Inf values:
    allvisit_tbl = allvisit_tbl[np.isfinite(allvisit_tbl['VHELIO']) &
                                np.isfinite(allvisit_tbl['VRELERR'])]
    allvisit_tbl = allvisit_tbl[(allvisit_tbl['VRELERR'] < 100.) &
                                (allvisit_tbl['VHELIO'] != -9999)]
    logger.log(1, "Filtered bad/NaN/-9999 data")
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

    allvisit_tbl = allvisit_tbl[np.isin(allvisit_tbl['APOGEE_ID'],
                                        allstar_tbl['APOGEE_ID'])]

    return allstar_tbl, allvisit_tbl
