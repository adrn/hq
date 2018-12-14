# Standard library
import time

# third party
import astropy.units as u
import numpy as np

# Package
from thejoker import JokerParams
from .log import log as logger

__all__ = ['Timer', 'config_to_jokerparams']

class Timer(object):

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, *args):
        self.time = self.elapsed()

    def reset(self):
        self.start = time.clock()

    def elapsed(self):
        return time.clock() - self.start


def config_to_jokerparams(config):
    P_min = u.Quantity(*config['hyperparams']['P_min'].split())
    P_max = u.Quantity(*config['hyperparams']['P_max'].split())

    if 'jitter' in config['hyperparams']:
        # jitter is fixed to some quantity, specified in config file
        jitter = u.Quantity(*config['hyperparams']['jitter'].split())
        logger.debug('Jitter is fixed to: {0:.2f}'.format(jitter))
        joker_pars = JokerParams(P_min=P_min, P_max=P_max,
                                 jitter=jitter)

    elif 'jitter_prior_mean' in config['hyperparams']:
        # jitter prior parameters are specified in config file
        jitter_mean = config['hyperparams']['jitter_prior_mean']
        jitter_stddev = config['hyperparams']['jitter_prior_stddev']
        jitter_unit = config['hyperparams']['jitter_prior_unit']
        logger.debug('Sampling in jitter with mean = {0:.2f} (stddev in '
                     'log(var) = {1:.2f}) [{2}]'
                     .format(np.sqrt(np.exp(jitter_mean)),
                             jitter_stddev, jitter_unit))
        joker_pars = JokerParams(P_min=P_min, P_max=P_max,
                                 jitter=(jitter_mean, jitter_stddev),
                                 jitter_unit=u.Unit(jitter_unit))

    else:
        joker_pars = JokerParams(P_min=P_min, P_max=P_max)

    return joker_pars
