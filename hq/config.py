import os
from os.path import expanduser, join

# Third-party
import astropy.units as u
import numpy as np

# Project
from thejoker import JokerParams
from .log import log as logger

__all__ = ['config_to_jokerparams']

HQ_CACHE_PATH = expanduser(os.environ.get("HQ_CACHE_PATH",
                                          join("~", ".hq")))

if not os.path.exists(HQ_CACHE_PATH):
    os.makedirs(HQ_CACHE_PATH, exist_ok=True)

logger.debug("Using cache path:\n\t {}\nSet the environment variable "
             "'HQ_CACHE_PATH' to change.".format(HQ_CACHE_PATH))


def config_to_jokerparams(config):
    P_min = u.Quantity(*config['hyperparams']['P_min'].split())
    P_max = u.Quantity(*config['hyperparams']['P_max'].split())
    kwargs = dict(P_min=P_min, P_max=P_max)

    if 'poly_trend' in config['hyperparams']:
        kwargs['poly_trend'] = config['hyperparams']['poly_trend']

    if 'jitter' in config['hyperparams']:
        # jitter is fixed to some quantity, specified in config file
        jitter = u.Quantity(*config['hyperparams']['jitter'].split())
        logger.debug('Jitter is fixed to: {0:.2f}'.format(jitter))
        kwargs['jitter'] = jitter

    elif 'jitter_prior_mean' in config['hyperparams']:
        # jitter prior parameters are specified in config file
        jitter_mean = config['hyperparams']['jitter_prior_mean']
        jitter_stddev = config['hyperparams']['jitter_prior_stddev']
        jitter_unit = config['hyperparams']['jitter_prior_unit']
        logger.debug('Sampling in jitter with mean = {0:.2f} (stddev in '
                     'log(var) = {1:.2f}) [{2}]'
                     .format(np.sqrt(np.exp(jitter_mean)),
                             jitter_stddev, jitter_unit))
        kwargs['jitter'] = (jitter_mean, jitter_stddev)
        kwargs['jitter_unit'] = u.Unit(jitter_unit)

    return JokerParams(**kwargs)
