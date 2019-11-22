# Standard library
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


class Config:

    def __init__(self, filename):
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

        # Some validation of settings:
        required = ['name', 'allstar_filename', 'allvisit_filename']
        for name in required:
            if not hasattr(self, name):
                raise ValueError(f"You must specify a config value for {name}")

    @property
    def cache_path(self):
        _path = os.path.join(HQ_CACHE_PATH, self.name)
        os.makedirs(_path, exist_ok=True)
        return _path

    def _load_data(self):
        allstar = fits.getdata(self.allstar_filename)
        allvisit = fits.getdata(self.allvisit_filename)
        return allstar, allvisit

    @property
    def allstar(self):
        try:
            d = self._allstar
        except AttributeError:
            self._allstar, self._allvisit = filter_alldata(
                *self._load_data(), min_nvisits=self.min_nvisits)
            d = self._allstar

        return d

    @property
    def allvisit(self):
        try:
            d = self._allvisit
        except AttributeError:
            self._allstar, self._allvisit = filter_alldata(
                *self._load_data(), min_nvisits=self.min_nvisits)
            d = self._allvisit

        return d
