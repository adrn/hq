import numpy as np
import astropy.units as u
from astropy.constants import G


def stellar_radius(logg, mass):
    return np.sqrt(G*mass / (10**logg*u.cm/u.s**2)).to(u.Rsun)


def period_at_surface(M1, logg, e, M2=0*u.Msun):
    R1 = np.sqrt(G*M1 / (10**logg * u.cm/u.s**2))
    q = M2 / M1
    P = 2*np.pi * (G*(M1+M2) / R1**3)**(-1/2) * (1-e)**(-3/2) * (1+q)**3
    return P.to(u.day)
