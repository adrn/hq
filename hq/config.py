# Standard library
from dataclasses import dataclass, fields
import os
import importlib.util as iu

# Third-party
from astropy.io import fits
import yaml

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
    max_prior_samples: int = 500_000_000
    prior_cache_file: str = ''
    requested_samples_per_star: int = 1024
    randomize_prior_order: bool = False

    # MCMC
    tune: int = 1000
    draws: int = 1000
    target_accept: float = 0.95

    def __init__(self, filename):
        filename = os.path.abspath(os.path.expanduser(filename))
        if not os.path.exists(filename):
            raise IOError(f"Config file {filename} does not exist.")

        with open(filename, 'r') as f:
            vals = yaml.safe_load(f.read())

        self._run_path, run_name = os.path.split(filename)
        self._load_validate(vals)

    def _load_validate(self, vals):
        # Validate types:
        for field in fields(self):
            default = None
            if field.name == 'prior_cache_file':
                default = (f'prior_samples_{self.n_prior_samples}'
                           f'_{self.name}.hdf5')

            val = vals.get(field.name, None)
            if val is None:
                val = default
            setattr(self, field.name, val)

            attr = getattr(self, field.name)
            if not isinstance(attr, field.type):
                msg = (f"Config field '{field.name}' has type {type(attr)}, "
                       f"but should be {field.type}")
                raise ValueError(msg)

        self.allstar_filename = os.path.abspath(
            os.path.expanduser(self.allstar_filename))
        self.allvisit_filename = os.path.abspath(
            os.path.expanduser(self.allvisit_filename))

        for name in ['allstar_filename', 'allvisit_filename']:
            val = getattr(self, name)
            if not os.path.exists(val):
                raise FileNotFoundError(f"File not found at {val}")

        # Normalize paths:
        if os.path.abspath(self.prior_file) != self.prior_file:
            self.prior_file = os.path.join(self.run_path, self.prior_file)

        if os.path.abspath(self.prior_cache_file) != self.prior_cache_file:
            self.prior_cache_file = os.path.join(self.run_path,
                                                 self.prior_cache_file)

    @classmethod
    def from_run_name(cls, name):
        return cls(os.path.join(HQ_CACHE_PATH, name, 'config.yml'))

    # ------------------------------------------------------------------------
    # Paths:

    @property
    def run_path(self):
        _path = self._run_path
        if _path is None:
            _path = os.path.join(HQ_CACHE_PATH, self.name)
        os.makedirs(_path, exist_ok=True)
        return _path

    @property
    def joker_results_path(self):
        return os.path.join(self.run_path, 'thejoker-samples.hdf5')

    @property
    def mcmc_results_path(self):
        return os.path.join(self.run_path, 'mcmc-samples.hdf5')

    @property
    def tasks_path(self):
        return os.path.join(self.run_path, 'tmp-tasks.hdf5')

    @property
    def metadata_path(self):
        return os.path.join(self.run_path, 'metadata.fits')

    @property
    def metadata_joker_path(self):
        return os.path.join(self.run_path, 'metadata-thejoker.fits')

    @property
    def metadata_mcmc_path(self):
        return os.path.join(self.run_path, 'metadata-mcmc.fits')

    # ------------------------------------------------------------------------
    # Data loading:

    def load_alldata(self):
        allstar = fits.getdata(self.allstar_filename)
        allvisit = fits.getdata(self.allvisit_filename)
        allstar, allvisit = filter_alldata(allstar, allvisit,
                                           min_nvisits=self.min_nvisits)
        return allstar, allvisit

    def get_prior(self, which=None):
        spec = iu.spec_from_file_location("prior", self.prior_file)
        user_prior = iu.module_from_spec(spec)
        spec.loader.exec_module(user_prior)
        if which == 'mcmc':
            return user_prior.prior_mcmc
        else:
            return user_prior.prior
