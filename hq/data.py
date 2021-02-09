# Third-party
from astropy.time import Time
import astropy.units as u
import numpy as np
from thejoker import RVData

from .log import logger


def get_rvdata(visits, err_column='VRELERR', dtype='f8', time_scale='tdb'):

    t = Time(visits['JD'].astype(dtype), format='jd', scale=time_scale)
    rv = visits['VHELIO'].astype(dtype) * u.km/u.s
    rv_err = visits[err_column]

    return RVData(t=t, rv=rv,
                  rv_err=rv_err.astype(dtype) * u.km/u.s)


def filter_data(allstar_tbl, allvisit_tbl,
                starflag_bits=None, aspcapflag_bits=None, rvflag_bits=None,
                min_nvisits=3):
    logger.log(1,
               f"Processing allstar ({len(allstar_tbl)} sources) and "
               f"allvisit ({len(allvisit_tbl)} visits) tables")

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
        # VERY_BRIGHT_NEIGHBOR, SUSPECT_RV_COMBINATION
        star_starflag_bits = [3, 16]
        visit_starflag_bits = star_starflag_bits

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

    # apply new (DR17) RV_FLAG masking:
    if rvflag_bits is None:
        rvflag_mask = allvisit_tbl['RV_FLAG'] == 0
    else:
        visit_rvflag_val = np.sum(2 ** np.array(rvflag_bits))
        rvflag_mask = (allstar_tbl['RVFLAG'] & visit_rvflag_val) == 0

    logger.log(1,
               f"Applying allvisit RVFLAG mask, filtered "
               f"{len(allvisit_tbl) - rvflag_mask.sum()} visits")

    # After quality and bitmask cut, figure out what APOGEE_IDs remain
    allvisit_tbl = allvisit_tbl[visit_starflag_mask & rvflag_mask]
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
