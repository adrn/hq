# Third-party
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import numpy as np
from thejoker.data import RVData


def get_alldata(allstar_path, allvisit_path,
                starflag_bits=None, aspcapflag_bits=None,
                min_nvisits=2):
    allvisit_tbl = fits.getdata(allvisit_path)
    allstar_tbl = fits.getdata(allstar_path)

    # Remove bad velocities / NaN / Inf values:
    allvisit_tbl = allvisit_tbl[np.isfinite(allvisit_tbl['VHELIO']) &
                                np.isfinite(allvisit_tbl['VRELERR'])]
    allvisit_tbl = allvisit_tbl[(allvisit_tbl['VRELERR'] < 100.) &
                                (allvisit_tbl['VHELIO'] != -9999)]

    if starflag_bits is None: # use deaults
        # LOW_SNR, PERSIST_HIGH, PERSIST_JUMP_POS, PERSIST_JUMP_NEG
        starflag_bits = [4, 9, 12, 13]

        # VERY_BRIGHT_NEIGHBOR, SUSPECT_RV_COMBINATION, SUSPECT_BROAD_LINES
        starflag_bits += [3, 16, 17]

    starflag_mask = np.sum(2 ** np.array(starflag_bits))
    allvisit_tbl = allvisit_tbl[(allvisit_tbl['STARFLAG'] & starflag_mask) == 0]

    # After quality and bitmask cut, figure out what APOGEE_IDs remain
    v_apogee_ids, counts = np.unique(allvisit_tbl['APOGEE_ID'],
                                     return_counts=True)
    star_mask = np.isin(allstar_tbl['APOGEE_ID'],
                        v_apogee_ids[counts >= min_nvisits])

    if aspcapflag_bits is None: # use defaults
        # TEFF_WARN, ROTATION_WARN, CHI2_WARN, STAR_BAD
        aspcapflag_bits = [0, 8, 10, 23]
    skip_mask = np.sum(2 ** np.array(aspcapflag_bits))
    star_mask &= ((allstar_tbl['ASPCAPFLAG'] & skip_mask) == 0)

    # Remove stars flagged with:
    # VERY_BRIGHT_NEIGHBOR, SUSPECT_RV_COMBINATION, SUSPECT_BROAD_LINES
    star_mask &= ((allstar_tbl['STARFLAG'] & starflag_mask) == 0)
    allstar_tbl = allstar_tbl[star_mask]

    # Only load visits for stars that we're loading
    allvisit_tbl = allvisit_tbl[np.isin(allvisit_tbl['APOGEE_ID'],
                                        allstar_tbl['APOGEE_ID'])]
    v_apogee_ids2 = np.unique(allvisit_tbl['APOGEE_ID'])
    star_mask2 = np.isin(allstar_tbl['APOGEE_ID'], v_apogee_ids2)

    allvisit_tbl = Table(allvisit_tbl)
    allstar_tbl = Table(allstar_tbl[star_mask2])

    return allstar_tbl, allvisit_tbl


def get_rvdata(visits):
    t = Time(visits['JD'], format='jd', scale='utc')
    rv = visits['VHELIO'] * u.km/u.s
    rv_err = visits['VRELERR'] * u.km/u.s
    return RVData(t=t, rv=rv, stddev=rv_err)
