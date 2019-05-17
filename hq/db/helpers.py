# Standard library
from os.path import join

# Third-party
import astropy.units as u
import numpy as np
from sqlalchemy import func
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound

# Project
from ..config import HQ_CACHE_PATH
from ..log import log as logger
from .model import JokerRun, AllStar, AllVisit, StarResult, AllVisitToAllStar

__all__ = ['get_run', 'paged_query']


def get_run(name, joker_params, n_requested_samples, session, overwrite=False):
    """Get a JokerRun row instance. Create one if it doesn't exist, otherwise
    just return the existing one.
    """

    # See if this run (by name) is in the database already, if so, grab that.
    try:
        run = session.query(JokerRun).filter(
            JokerRun.name == name).one()
        logger.info("JokerRun '{0}' already found in database"
                    .format(name))

    except NoResultFound:
        run = None

    except MultipleResultsFound:
        raise MultipleResultsFound("Multiple JokerRun rows found for name '{0}'"
                                   .format(name))

    if run is not None:
        if overwrite:
            session.query(StarResult)\
                   .filter(StarResult.jokerrun_id == run.id)\
                   .delete()
            session.commit()
            session.delete(run)
            session.commit()

        else:
            return run

    # If we've gotten here, this run doesn't exist in the database yet, so
    # create it using the parameters read from the config file.
    logger.info("JokerRun '{0}' not found in database, creating entry..."
                .format(name))

    # Create a JokerRun for this run
    run = JokerRun()
    run.name = name
    run.P_min = joker_params.P_min
    run.P_max = joker_params.P_max
    run.requested_samples_per_star = n_requested_samples

    # TODO:
    run.prior_samples_file = ''
    run.max_prior_samples = 0

    if hasattr(joker_params.jitter, 'unit'):
        # jitter is fixed to some quantity, specified in config file
        run.jitter = joker_params.jitter
        logger.debug('Jitter is fixed to: {0:.2f}'.format(run.jitter))

    elif joker_params.jitter == 0 * u.m/u.s:
        # no jitter is specified, assume no jitter
        run.jitter = 0. * u.m/u.s
        logger.debug('No jitter.')

    else:
        # jitter prior parameters are specified in config file
        run.jitter_mean = joker_params.jitter[0]
        run.jitter_stddev = joker_params.jitter[1]
        run.jitter_unit = str(joker_params._jitter_unit)
        logger.debug('Sampling in jitter with mean = {0:.2f} (stddev in '
                     'log(var) = {1:.2f}) [{2}]'
                     .format(np.sqrt(np.exp(run.jitter_mean)),
                             run.jitter_stddev, run.jitter_unit))

    run.poly_trend = joker_params.poly_trend

    # Get all stars with >=3 visits
    q = session.query(AllStar).join(AllVisitToAllStar, AllVisit)\
                              .group_by(AllStar.apstar_id)\
                              .having(func.count(AllVisit.id) >= 3)

    stars = q.all()

    run.stars = stars
    session.add(run)
    session.commit()

    return run


def paged_query(query, page_size=1000):
    """
    """
    n_tot = query.count()

    n_pages = n_tot // page_size
    if n_tot % page_size:
        n_pages += 1

    for page in range(n_pages):
        q = query.limit(page_size)

        if page:
            q = q.offset(page*page_size)

        yield q
