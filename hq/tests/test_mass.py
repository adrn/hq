# Third-party
from astropy.utils.data import get_pkg_data_filename
from astropy.io import ascii
import numpy as np

# Project
from ..mass import get_martig_vec, M

def test_martig_mass():

    martig_tbl = ascii.read(get_pkg_data_filename('martig_tbl.txt'))
    Merr = 0.25 # Msun

    for row in martig_tbl:
        x = get_martig_vec(row['Teff'], row['logg'], row['[M/H]'], row['[C/M]'], row['[N/M]'])
        mass = M.dot(x).dot(x)
        assert np.isclose(mass, row['Mout'], atol=Merr)
