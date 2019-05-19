# Third-party
from astropy.time import Time
import astropy.units as u
from thejoker.data import RVData


def get_rvdata(visits):
    t = Time(visits['JD'], format='jd', scale='utc')
    rv = visits['VHELIO'] * u.km/u.s
    rv_err = visits['VRELERR'] * u.km/u.s
    return RVData(t=t, rv=rv, stddev=rv_err)
