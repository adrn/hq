from os import path
import astropy.units as u

import numpy as np
import h5py

from thejoker import JokerSamples

from twoface.config import TWOFACE_CACHE_PATH
from twoface.db import (db_connect, AllStar, AllVisit, AllVisitToAllStar, RedClump,
                        StarResult, Status, JokerRun, initialize_db)
from twoface import unimodal_P

samples_file = path.join(TWOFACE_CACHE_PATH, 'apogee-jitter.hdf5')
Session, _ = db_connect(path.join(TWOFACE_CACHE_PATH, 'apogee.sqlite'))
session = Session()

with h5py.File(samples_file) as f:
    for i, key in enumerate(f):
        g = f[key]

        if 'ecc' in g:
            try:
                g['e'] = g['ecc'][:]
            except RuntimeError:
                print(key)
                raise

            g['M0'] = g['phi0'][:]
            for k, v in dict(g['phi0'].attrs).items():
                g['M0'].attrs[k] = v

            del g['ecc'], g['phi0']
        else:
            continue

        if i % 1000 == 0:
            print(i)

with h5py.File(samples_file) as f:
    for key in f:
        if 't0_bmjd' not in f[key].attrs:
            f[key].attrs['t0_bmjd'] = 0.

# update the statuses of stars that need more samples:
results = session.query(StarResult).join(AllStar, JokerRun, Status)\
                 .filter(JokerRun.name == 'apogee-jitter')\
                 .filter(Status.id.in_([1, 2])).all()

for i, result in enumerate(results):
    star = result.star
    data = star.apogeervdata()

    with h5py.File(samples_file) as f:
        samples = JokerSamples.from_hdf5(f[star.apogee_id])

    if unimodal_P(samples, data):
        # Multiple samples were returned, but they look unimodal
        result.status_id = 2 # needs mcmc

    else:
        # Multiple samples were returned, but not enough to satisfy the
        # number requested in the config file
        result.status_id = 1 # needs more samples

    if i % 100 == 0:
        print(i)
        session.commit()

session.commit()
