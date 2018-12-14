# Standard library
from os.path import join

# Third-party
import astropy.units as u
import numpy as np
from sqlalchemy import func
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound

# Project
from twoface.log import log as logger
from twoface.db import (JokerRun, AllStar, AllVisit, StarResult,
                        AllVisitToAllStar)
from twoface.config import TWOFACE_CACHE_PATH

__all__ = ['get_run']


def get_run(config, session, overwrite=False):
    """Get a JokerRun row instance. Create one if it doesn't exist, otherwise
    just return the existing one.
    """

    # See if this run (by name) is in the database already, if so, grab that.
    try:
        run = session.query(JokerRun).filter(
            JokerRun.name == config['name']).one()
        logger.info("JokerRun '{0}' already found in database"
                    .format(config['name']))

    except NoResultFound:
        run = None

    except MultipleResultsFound:
        raise MultipleResultsFound("Multiple JokerRun rows found for name '{0}'"
                                   .format(config['name']))

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
                .format(config['name']))

    # Create a JokerRun for this run
    run = JokerRun()
    run.config_file = config['config_file']
    run.name = config['name']
    run.P_min = u.Quantity(*config['hyperparams']['P_min'].split())
    run.P_max = u.Quantity(*config['hyperparams']['P_max'].split())
    run.requested_samples_per_star = int(
        config['hyperparams']['requested_samples_per_star'])
    run.max_prior_samples = int(config['prior']['max_samples'])
    run.prior_samples_file = join(TWOFACE_CACHE_PATH,
                                  config['prior']['samples_file'])

    if 'jitter' in config['hyperparams']:
        # jitter is fixed to some quantity, specified in config file
        run.jitter = u.Quantity(*config['hyperparams']['jitter'].split())
        logger.debug('Jitter is fixed to: {0:.2f}'.format(run.jitter))

    elif 'jitter_prior_mean' in config['hyperparams']:
        # jitter prior parameters are specified in config file
        run.jitter_mean = config['hyperparams']['jitter_prior_mean']
        run.jitter_stddev = config['hyperparams']['jitter_prior_stddev']
        run.jitter_unit = config['hyperparams']['jitter_prior_unit']
        logger.debug('Sampling in jitter with mean = {0:.2f} (stddev in '
                     'log(var) = {1:.2f}) [{2}]'
                     .format(np.sqrt(np.exp(run.jitter_mean)),
                             run.jitter_stddev, run.jitter_unit))

    else:
        # no jitter is specified, assume no jitter
        run.jitter = 0. * u.m/u.s
        logger.debug('No jitter.')

    # Get all stars with >=3 visits
    q = session.query(AllStar).join(AllVisitToAllStar, AllVisit)\
                              .group_by(AllStar.apstar_id)\
                              .having(func.count(AllVisit.id) >= 3)

    stars = q.all()

    run.stars = stars
    session.add(run)
    session.commit()

    return run
