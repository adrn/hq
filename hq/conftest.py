# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure. It needs to live inside the package in order for it to
# get picked up when running the tests inside an interpreter using
# packagename.test

import os

from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

# pytest fixtures:


def pytest_configure(config):
    config.option.astropy_header = True

    # Customize the following lines to add/remove entries from the list of
    # packages for which version numbers are displayed when running the tests.
    PYTEST_HEADER_MODULES["thejoker"] = "thejoker"
    PYTEST_HEADER_MODULES["twobody"] = "twobody"
    PYTEST_HEADER_MODULES["schwimmbad"] = "schwimmbad"

    from . import __version__

    packagename = os.path.basename(os.path.dirname(__file__))
    TESTED_VERSIONS[packagename] = __version__
