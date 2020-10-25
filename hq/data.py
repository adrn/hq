# Third-party
from astropy.time import Time
import astropy.units as u
import numpy as np
from .log import logger


def get_new_err(visits, a, b, s):
    err = visits['VRELERR']
    snr = visits['SNR']
    return np.sqrt(s**2 + err**2 + a*snr**b)


def get_new_err_nidever(visits):
    err = visits['VRELERR']
    var = (3.5*err**1.2)**2 + 0.072**2
    return np.sqrt(var)


def get_rvdata(visits, apply_error_calibration=True, float64=True,
               assume_tcb=False):
    from thejoker import RVData
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
        # rv_err = get_new_err(visits,
        #                      a=145.869, b=-2.8215, s=0.168)

        # See email from Nidever, apogee2-rvvar 1421
        rv_err = get_new_err_nidever(visits)

    else:
        rv_err = visits['VRELERR']

    return RVData(t=t, rv=rv, rv_err=rv_err.astype(dtype) * u.km/u.s)


def filter_alldata(allstar_tbl, allvisit_tbl,
                   starflag_bits=None, aspcapflag_bits=None,
                   min_nvisits=3):
    logger.log(1,
               f"Opened allstar ({len(allstar_tbl)} sources) and "
               f"allvisit ({len(allvisit_tbl)} visits)")

    # Remove bad velocities / NaN / Inf values:
    bad_visit_mask = (
        np.isfinite(allvisit_tbl['VHELIO']) &
        np.isfinite(allvisit_tbl['VRELERR']) &
        (allvisit_tbl['VRELERR'] < 100.) &
        (allvisit_tbl['VHELIO'] != -9999) &
        (np.abs(allvisit_tbl['VHELIO']) < 500.)
    )
    logger.log(1,
               f"Filtered {len(bad_visit_mask) - bad_visit_mask.sum()} "
               "bad/NaN/-9999 visits")
    allvisit_tbl = allvisit_tbl[bad_visit_mask]

    if starflag_bits is None:  # use defaults
        # VERY_BRIGHT_NEIGHBOR, SUSPECT_RV_COMBINATION, SUSPECT_BROAD_LINES
        star_starflag_bits = [3, 16]  # 17

        # LOW_SNR, PERSIST_HIGH, PERSIST_JUMP_POS, PERSIST_JUMP_NEG
        visit_starflag_bits = star_starflag_bits + [4, 9, 12, 13]

    star_starflag_val = np.sum(2 ** np.array(star_starflag_bits))
    visit_starflag_val = np.sum(2 ** np.array(visit_starflag_bits))
    star_starflag_mask = (allstar_tbl['STARFLAG'] & star_starflag_val) == 0
    visit_starflag_mask = (allvisit_tbl['STARFLAG'] & visit_starflag_val) == 0

    logger.log(1,
               f"Using allstar STARFLAG bitmask {star_starflag_val}), "
               f"filtered {len(allstar_tbl) - star_starflag_mask.sum()} "
               "sources")
    logger.log(1,
               f"Using allvisit STARFLAG bitmask {visit_starflag_val}), "
               f"filtered {len(allvisit_tbl) - visit_starflag_mask.sum()} "
               "visits")

    # After quality and bitmask cut, figure out what APOGEE_IDs remain
    allvisit_tbl = allvisit_tbl[visit_starflag_mask]
    v_apogee_ids, counts = np.unique(allvisit_tbl['APOGEE_ID'],
                                     return_counts=True)
    allstar_visit_mask = np.isin(allstar_tbl['APOGEE_ID'],
                                 v_apogee_ids[counts >= min_nvisits])
    logger.log(1,
               f"Keeping only sources with > {min_nvisits} visits: filtered "
               f"{len(allstar_visit_mask) - allstar_visit_mask.sum()} sources")

    if aspcapflag_bits is None:  # use defaults
        # TEFF_BAD, LOGG_BAD, VMICRO_BAD, ROTATION_BAD, VSINI_BAD
        aspcapflag_bits = [16, 17, 18, 26, 30]

    # So that this can be run on preliminary / pre-ASPCAP data:
    if 'ASPCAPFLAG' in allstar_tbl.dtype.names:
        aspcapflag_val = np.sum(2 ** np.array(aspcapflag_bits))
        aspcapflag_mask = (allstar_tbl['ASPCAPFLAG'] & aspcapflag_val) == 0
        logger.log(1, f"Using allstar ASPCAPFLAG bitmask {aspcapflag_val}), "
                      f"filtered {len(allstar_tbl) - aspcapflag_mask.sum()}")
    else:
        aspcapflag_mask = True

    allstar_tbl = allstar_tbl[allstar_visit_mask &
                              star_starflag_mask &
                              aspcapflag_mask]

    # Only load visits for stars that we're loading
    allvisit_tbl = allvisit_tbl[np.isin(allvisit_tbl['APOGEE_ID'],
                                        allstar_tbl['APOGEE_ID'])]
    v_apogee_ids2 = np.unique(allvisit_tbl['APOGEE_ID'])
    star_mask2 = np.isin(allstar_tbl['APOGEE_ID'], v_apogee_ids2)
    allstar_tbl = allstar_tbl[star_mask2]

    _, idx = np.unique(allstar_tbl['APOGEE_ID'], return_index=True)
    allstar_tbl = allstar_tbl[idx]

    logger.log(1,
               f"{len(allstar_tbl)} unique allstar rows left after STARFLAG, "
               "ASPCAPFLAG, and min. visit filtering")

    allvisit_tbl = allvisit_tbl[np.isin(allvisit_tbl['APOGEE_ID'],
                                        allstar_tbl['APOGEE_ID'])]

    return allstar_tbl, allvisit_tbl
