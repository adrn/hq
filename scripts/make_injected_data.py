"""
Generate simulated radial velocity data (as a simulated allVisit file) for APOGEE sources that have already been run with The Joker.
"""

# Standard library
import os
import socket
import sys

# Third-party
import astropy.table as at
import astropy.units as u
import numpy as np
from thejoker.multiproc_helpers import batch_tasks

# Project
from hq.log import logger
from hq.config import Config
from hq.data import get_rvdata
from hq.physics_helpers import period_at_surface
from hq.samples_analysis import extract_MAP_sample


def worker(task):
    # allstar table must also have columns from metadata
    allstar, allvisit, worker_id, c, prior = task

    rnd = np.random.Generator(np.random.PCG64(worker_id))

    # prior = c.get_prior()
    logger.debug(f"Worker {worker_id} on node {socket.gethostname()}: "
                 f"{len(allstar)} stars left to process")

    for star in allstar:
        apogee_id = star['APOGEE_ID']
        visits = allvisit[allvisit['APOGEE_ID'] == apogee_id]
        data = get_rvdata(visits)

        logger.debug(f"Worker {worker_id}: Making {len(data)} visits")

        MAP_sample = extract_MAP_sample(star)
        MAP_orbit = MAP_sample.get_orbit()

        data = get_rvdata(allvisit[allvisit['APOGEE_ID'] == star['APOGEE_ID']])
        tmp_rv = data.rv - MAP_orbit.radial_velocity(data.t)

        P_min = period_at_surface(0.8*u.Msun, star['LOGG'], e=0.)
        ecc = rnd.uniform()

        tmp_samples = prior.sample(size=128, generate_linear=True)
        tmp_samples = tmp_samples[tmp_samples['P'] > P_min]

        sample = tmp_samples[0]
        sample.tbl['e'] = ecc

        orbit = sample.get_orbit()
        orbit.t0 = data.t0

        new_rv = tmp_rv + orbit.radial_velocity(data.t)
        new_rv = new_rv.to_value(u.km/u.s)

        allvisit['VHELIO'][allvisit['APOGEE_ID'] == apogee_id] = new_rv

    # result = {'allvisit': allvisit,
    #           'hostname': socket.gethostname(),
    #           'worker_id': worker_id}
    # return result

    return at.Table(allvisit)


def main(run_name, pool, overwrite=False, limit=None):
    c = Config.from_run_name(run_name)

    if not os.path.exists(c.prior_cache_file):
        raise IOError(f"Prior cache file {c.prior_cache_file} does not exist! "
                      "Did you run make_prior_cache.py?")

    # Get data files out of config file:
    logger.debug("Loading data...")
    allstar, allvisit = c.load_alldata()

    # Only keep stars with measured LOGG (to estimate surface period):
    allstar = allstar[allstar['LOGG'] > -0.5]

    # Only keep some columns of allvisit file:
    allvisit = at.Table(allvisit)['VISIT_ID', 'APOGEE_ID',
                                  'JD', 'VHELIO', 'VRELERR', 'STARFLAG']

    # Get metadata file from previous full run of HQ:
    logger.debug("Loading metadata...")
    metadata = at.QTable.read(c.metadata_path)

    # Combine allstar and metadata/results data:
    allstar = at.join(metadata, allstar, keys='APOGEE_ID')

    logger.debug("Preparing tasks...")
    if len(allstar) > 10 * pool.size:
        n_tasks = min(16 * pool.size, len(allstar))
    else:
        n_tasks = pool.size

    prior = c.get_prior()

    tasks = []
    for (i1, i2), _ in batch_tasks(len(allstar), n_tasks):
        stars = allstar[i1:i2]
        visits = allvisit[np.isin(allvisit['APOGEE_ID'], stars['APOGEE_ID'])]
        tasks.append([stars, visits, i1, c, prior])

    logger.info(f'Done preparing tasks: split into {len(tasks)} task chunks')
    new_allvisit_chunks = []
    for allvisit_chunk in pool.map(worker, tasks):
        new_allvisit_chunks.append(allvisit_chunk)

    new_allvisit = at.vstack(new_allvisit_chunks)
    new_allvisit_filename = os.path.join(c.run_path, 'allVisit-injected.fits')
    new_allvisit.write(new_allvisit_filename)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    from hq.script_helpers import get_parser

    # Define parser object
    parser = get_parser(description='Make simulated radial velocity data',
                        loggers=[logger])

    args = parser.parse_args()

    with threadpool_limits(limits=1, user_api='blas'):
        with args.Pool(**args.Pool_kwargs) as pool:
            main(run_name=args.run_name, pool=pool, overwrite=args.overwrite)

    sys.exit(0)
