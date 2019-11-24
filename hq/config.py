# Standard library
from dataclasses import dataclass, fields
import os
import importlib.util as iu

# Third-party
from astropy.io import fits

# Project
from .log import logger
from .data import filter_alldata

__all__ = ['HQ_CACHE_PATH', 'Config']

HQ_CACHE_PATH = os.path.expanduser(
    os.environ.get("HQ_CACHE_PATH", os.path.join("~", ".hq")))

if not os.path.exists(HQ_CACHE_PATH):
    os.makedirs(HQ_CACHE_PATH, exist_ok=True)

logger.debug("Using cache path:\n\t {}\nSet the environment variable "
             "'HQ_CACHE_PATH' to change.".format(HQ_CACHE_PATH))


@dataclass
class Config:
    name: str = ''
    allstar_filename: str = ''
    allvisit_filename: str = ''
    min_nvisits: int = 3

    # The Joker
    prior_file: str = 'prior.py'
    n_prior_samples: int = 500_000_000
    prior_cache_file: str = ''
    requested_samples_per_star: int = 1024
    randomize_prior_order: bool = False

    # MCMC
    tune: int = 1000
    draws: int = 1000
    target_accept: float = 0.95

    def __init__(self, filename):
        self.name = filename

        filename = os.path.abspath(os.path.expanduser(filename))
        if not os.path.exists(filename):
            raise IOError(f"Config file {filename} does not exist.")

        spec = iu.spec_from_file_location("config", filename)
        user_config = iu.module_from_spec(spec)
        spec.loader.exec_module(user_config)

        for name in dir(user_config):
            if name.startswith('_'):
                continue
            setattr(self, name, getattr(user_config, name))

        self._validate()

    def _validate(self):
        # Validate types:
        for field in fields(self):
            attr = getattr(self, field.name)
            if not isinstance(attr, field.type):
                msg = (f"Config field '{field.name}' has type {type(attr)}, "
                       f"but should be {field.type}")
                raise ValueError(msg)

        # Some validation of values too:
        required = ['name', 'allstar_filename', 'allvisit_filename']
        for name in required:
            if not hasattr(self, name):
                raise ValueError(f"You must specify a config value for {name}")

        # Normalize paths:
        if os.path.abspath(self.prior_file) != self.prior_file:
            self.prior_file = os.path.join(self.run_path, self.prior_file)

        if self.prior_cache_file is None:
            self.prior_cache_file = (f'prior_samples_{self.n_prior_samples}'
                                     f'_{self.name}.hdf5')

        if os.path.abspath(self.prior_cache_file) != self.prior_cache_file:
            self.prior_cache_file = os.path.join(self.run_path,
                                                 self.prior_cache_file)

    @classmethod
    def from_run_name(cls, name):
        return cls(os.path.join(HQ_CACHE_PATH, name, 'config.py'))

    # ------------------------------------------------------------------------
    # Paths:

    @property
    def run_path(self):
        _path = os.path.join(HQ_CACHE_PATH, self.name)
        os.makedirs(_path, exist_ok=True)
        return _path

    @property
    def joker_results_path(self):
        return os.path.join(self.run_path, 'thejoker-samples.hdf5')

    @property
    def tasks_path(self):
        return os.path.join(self.run_path, 'tmp-tasks.hdf5')

    @property
    def metadata_path(self):
        return os.path.join(self.run_path, 'metadata.fits')

    # ------------------------------------------------------------------------
    # Data loading:

    def load_alldata(self):
        allstar = fits.getdata(self.allstar_filename)
        allvisit = fits.getdata(self.allvisit_filename)
        allstar, allvisit = filter_alldata(allstar, allvisit,
                                           min_nvisits=self.min_nvisits)
        return allstar, allvisit

    def get_prior(self):
        spec = iu.spec_from_file_location("prior", self.prior_file)
        user_prior = iu.module_from_spec(spec)
        spec.loader.exec_module(user_prior)
        return user_prior.prior
