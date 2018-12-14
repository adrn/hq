# Standard library
from os import path
import time

# Third-party
import astropy.units as u
import numpy as np
from schwimmbad import choose_pool
from sqlalchemy import func
from thejoker.log import log as joker_logger
from thejoker.sampler import TheJoker, JokerParams

# Project
from twoface.log import log as logger
from twoface.db import db_connect
from twoface.db import AllStar, AllVisit
from twoface.config import TWOFACE_CACHE_PATH


def main(pool):
    seed = 42

    db_path = path.join(TWOFACE_CACHE_PATH, 'apogee.sqlite')
    logger.debug("Connecting to sqlite database at '{0}'".format(db_path))
    Session, engine = db_connect(database_path=db_path,
                                 ensure_db_exists=False)
    session = Session()

    apogee_id = '2M04171719+4724006' # needs more prior samples
    # apogee_id = '2M01320817+0120301' # needs mcmc
    # apogee_id = '2M02123870+4942289' # completed
    star = session.query(AllStar).filter(AllStar.apogee_id == apogee_id)\
                  .limit(1).one()
    data = star.apogeervdata()

    rnd = np.random.RandomState(seed=seed)
    params = JokerParams(P_min=8*u.day, P_max=32768*u.day)
    joker = TheJoker(params, random_state=rnd, pool=pool)

    prior_cache_file = path.join(TWOFACE_CACHE_PATH,
                                 'P8-32768_prior_samples.hdf5')

    print("Pool size: {0}".format(pool.size))

    n_iter = 4
    for max_prior_samples in 2 ** np.arange(7, 25+1, 2):
        t0 = time.time()
        for k in range(n_iter):
            try:
                samples = joker.rejection_sample(
                    data=data, prior_cache_file=prior_cache_file,
                    n_prior_samples=max_prior_samples)

            except Exception as e:
                logger.warning("\t Failed sampling for star {0} \n Error: {1}"
                               .format(star.apogee_id, str(e)))
                continue

        dt = (time.time() - t0) / n_iter
        print("{0}, {1:.3f}".format(max_prior_samples, dt))

    pool.close()
    session.close()


if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    args = parser.parse_args()

    for l in [joker_logger, logger]:
        l.setLevel(logging.DEBUG)

    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)
    pool = choose_pool(**pool_kwargs)

    main(pool=pool)
