# Standard library
from dataclasses import dataclass, fields
import os
import importlib.util as iu

# Third-party
from astropy.io import fits
import astropy.table as at
import yaml

# Project
from .log import logger
from .data import filter_alldata, get_rvdata

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

    # Calibrated visit RV uncertainties
    visit_error_filename: str = ''
    visit_error_colname: str = 'CALIB_VERR'

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

        self._cache = {}
        if (self.visit_error_filename is not None
                and len(self.visit_error_filename) > 0):
            self._cache['visit_err_tbl'] = fits.getdata(
                self.visit_error_filename)

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

    def dump_cache(self):
        """Purge any cached data tables"""
        self._cache = {}

    @property
    def allstar(self):
        if 'allstar' not in self._cache:
            (self._cache['allstar'],
             self._cache['allvisit']) = self.load_alldata()
        return self._cache['allstar']

    @property
    def allvisit(self):
        if 'allvisit' not in self._cache:
            (self._cache['allstar'],
             self._cache['allvisit']) = self.load_alldata()
        return self._cache['allvisit']

    def load_alldata(self):
        allstar = fits.getdata(self.allstar_filename)
        allvisit = fits.getdata(self.allvisit_filename)
        allstar, allvisit = filter_alldata(allstar, allvisit,
                                           min_nvisits=self.min_nvisits)

        if self._visit_err_tbl is not None:
            allvisit = at.join(
                at.Table(allvisit),
                at.Table(self._visit_err_tbl),
                # keys='VISITID', # TODO - HACK FOR DR17 alpha!
                keys=('PLATE', 'MJD', 'FIBERID'))

            # TODO - HACK FOR DR17 alpha!
            # allvisit = at.unique(allvisit, keys='VISITID')
            allvisit = at.unique(allvisit, keys=('PLATE', 'MJD', 'FIBERID'))

        return allstar, allvisit

    def get_star_data(self, apogee_id):
        visits = self.allvisit[self.allvisit['APOGEE_ID'] == apogee_id]
        if self._visit_err_tbl is not None:
            err_column = self.visit_error_colname
        else:
            err_column = 'VRELERR'
        return get_rvdata(visits, err_column=err_column)

    def get_prior(self, which=None):
        spec = iu.spec_from_file_location("prior", self.prior_file)
        user_prior = iu.module_from_spec(spec)
        spec.loader.exec_module(user_prior)
        if which == 'mcmc':
            return user_prior.prior_mcmc
        else:
            return user_prior.prior

    # Special methods
    def __getstate__(self):
        """Ensure that the cache does not get pickled with the object"""
        state = {k: v for k, v in self.__dict__.items() if 'cache' not in k}
        return state.copy()
