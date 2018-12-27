# Standard library
from os import path
import time

# Third-party
from schwimmbad import choose_pool
from thejoker.log import log as joker_logger
from thejoker.sampler import TheJoker

# Project
from hq.log import log as logger
from hq.db import db_connect, AllStar, JokerRun
from hq.config import HQ_CACHE_PATH


def main(pool):
    db_path = path.join(HQ_CACHE_PATH, 'apogee.sqlite')
    logger.debug("Connecting to sqlite database at '{0}'".format(db_path))
    Session, engine = db_connect(database_path=db_path,
                                 ensure_db_exists=False)
    session = Session()

    ids = ['2M00000002+7417074',
           '2M00000032+5737103',
           '2M00000032+5737103',
           '2M00000068+5710233',
           '2M00000222+5625359',
           '2M00000233+1452324',
           '2M00000446+5854329',
           '2M00000535+1504343',
           '2M00000546+6152107',
           '2M00000662+7528598']

    stars = session.query(AllStar).filter(AllStar.apogee_id.in_(ids)).all()
    run = session.query(JokerRun).limit(1).one()
    params = run.get_joker_params()
    joker = TheJoker(params, pool=pool)

    time0 = time.time()
    for star in stars:
        logger.log(1, "Starting star '{0}'".format(star.apogee_id))
        _t0 = time.time()

        data = star.get_rvdata()
        logger.log(1, "\t {0} visits loaded ({1:.2f} seconds)"
                   .format(len(data.rv), time.time()-t0))
        try:
            samples, ln_prior = joker.iterative_rejection_sample(
                data=data, n_requested_samples=run.requested_samples_per_star,
                prior_cache_file=run.prior_samples_file,
                n_prior_samples=run.max_prior_samples, return_logprobs=True)

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
        l.setLevel(logging.DEBUG)

    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)
    pool = choose_pool(**pool_kwargs)

    main(pool=pool)
