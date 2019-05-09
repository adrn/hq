# Standard library
from os import path
import time

# Third-party
import astropy.units as u
from schwimmbad import choose_pool
from thejoker.log import log as joker_logger
from thejoker.sampler import TheJoker, JokerParams

# Project
from hq.log import log as logger
from hq.db import db_connect, AllStar, AllVisitToAllStar, AllVisit
from hq.config import HQ_CACHE_PATH
from hq.sample_prior import make_prior_cache


def main(pool):
    db_path = path.join(HQ_CACHE_PATH, 'apogee.sqlite')
    logger.debug("Connecting to sqlite database at '{0}'".format(db_path))
    Session, engine = db_connect(database_path=db_path,
                                 ensure_db_exists=False)
    session = Session()

    stars = session.query(AllStar).join(AllVisitToAllStar, AllVisit)\
                   .filter(AllStar.apogee_id == '2M17162131-2451161').all()
    params = JokerParams(P_min=1*u.day, P_max=32768*u.day,
                         poly_trend=3, jitter=150*u.m/u.s)
    joker = TheJoker(params, pool=pool)

    prior_filename = path.join(HQ_CACHE_PATH, 'test_prior_samples.hdf5')
    if not path.exists(prior_filename):
        make_prior_cache(prior_filename, joker, nsamples=2**24,
                         batch_size=2**20)

    time0 = time.time()
    for star in stars:
        logger.log(1, "Starting star '{0}'".format(star.apogee_id))
        _t0 = time.time()

        data = star.get_rvdata()
        logger.log(1, "\t {0} visits loaded ({1:.2f} seconds)"
                   .format(len(data.rv), time.time()-_t0))
        try:
            _ = joker.rejection_sample(
                data=data, prior_cache_file=prior_filename,
                return_logprobs=True)

        except Exception as e:
            logger.warning("\t Failed sampling for star {0} \n Error: {1}"
                           .format(star.apogee_id, str(e)))
            continue

        logger.debug("\t done sampling - {0} raw samples returned "
                     "({1:.2f} seconds)".format(len(samples),
                                                time.time()-_t0))

        # For now, it's sufficient to write the run results to an HDF5 file
        n_raw_samples = len(samples)

        logger.debug("...done with star {0}: {1} visits, {2} samples returned "
                     " ({3:.2f} seconds)"
                     .format(star.apogee_id, len(data.rv), n_raw_samples,
                             time.time()-_t0))

    print("Pool: {0}, pool size: {1}, time: {2}"
          .format(pool.__class__.__name__, pool.size, time.time() - time0))

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
        l.setLevel(1)

    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)
    pool = choose_pool(**pool_kwargs)

    main(pool=pool)
