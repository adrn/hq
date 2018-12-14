import os
from os.path import expanduser, join

# Project
from .log import log as logger

TWOFACE_CACHE_PATH = expanduser(os.environ.get("TWOFACE_CACHE_PATH",
                                               join("~", ".twoface")))

if not os.path.exists(TWOFACE_CACHE_PATH):
    os.makedirs(TWOFACE_CACHE_PATH, exist_ok=True)

logger.debug("Using cache path:\n\t {}\nSet 'TWOFACE_CACHE_PATH' to change.")
