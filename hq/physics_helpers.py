import numpy as np
import astropy.units as u
from astropy.constants import G


def stellar_radius(logg, mass):
    return np.sqrt(G*mass / (10**logg*u.cm/u.s**2)).to(u.Rsun)


def period_at_surface(M1, logg, e, M2=0*u.Msun):
    R1 = np.sqrt(G*M1 / (10**logg * u.cm/u.s**2))
    q = M2 / M1
    P = 2*np.pi * (G*(M1+M2) / R1**3)**(-1/2) * (1-e)**(-3/2) * (1+q)**1.5
    return P.to(u.day)


def fast_mf(P, K, e):
    """Binary mass function."""
    mf_circ = P * K**3 / (2*np.pi * G)
    return mf_circ.to(u.Msun) * (1 - e**2)**1.5


def fast_m2_min(m1, mf):
    return (2*mf + np.power(2,2/3)*
      np.power(mf*(27*pow(m1,2) + 18*m1*mf + 2*np.power(mf,2) +
          3*np.sqrt(3)*np.sqrt(np.power(m1,3)*(27*m1 + 4*mf))),1/3) +
     (2*mf*(6*m1 + mf))/
      np.power((27*np.power(m1,2)*mf)/2. + 9*m1*np.power(mf,2) + np.power(mf,3) +
        (3*np.sqrt(3)*mf*np.sqrt(np.power(m1,3)*(27*m1 + 4*mf)))/2.,1/3))/6.
