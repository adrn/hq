# Third-party
from astropy.time import Time
import astropy.units as u
import numpy as np
from thejoker.data import RVData


def get_new_err(visits, a, b, s):
    err = visits['VRELERR']
    snr = visits['SNR']
    return np.sqrt(s**2 + err**2 + a*snr**b)


def get_rvdata(visits, apply_error_calibration=True, float64=True):
    if float64:
        dtype = 'f8'
    else:
        dtype = 'f4'

    t = Time(visits['JD'].astype(dtype), format='jd', scale='utc')
    rv = visits['VHELIO'].astype(dtype) * u.km/u.s

    if apply_error_calibration:
        # see notebook: Calibrate-visit-rv-err.ipynb
        rv_err = get_new_err(visits,
                             a=145.869, b=-2.8215, s=0.168)
    else:
        rv_err = visits['VRELERR']

    return RVData(t=t, rv=rv, stddev=rv_err.astype(dtype) * u.km/u.s)
