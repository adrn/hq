# Standard library
from os import path, unlink

# Third-party
from astropy.utils.data import get_pkg_data_filename
import pytest
import yaml

# Project
from ...config import HQ_CACHE_PATH
from ..connect import db_connect, session_from_config


def test_db_connect():
    db_path = path.join(HQ_CACHE_PATH, 'test.sqlite')

    with pytest.raises(IOError):
        db_connect(db_path)

    _ = db_connect(db_path, ensure_db_exists=True)
    assert path.exists(db_path)
    unlink(db_path)


def test_session_from_config():
    config_file = get_pkg_data_filename('travis_db.yml')

    with open(config_file, 'r') as f:
        config = yaml.load(f.read())
    db_path = path.join(HQ_CACHE_PATH, config['database_file'])

    with pytest.raises(IOError):
        session_from_config(config_file)

    _ = session_from_config(config_file, ensure_db_exists=True)
    assert path.exists(db_path)
    unlink(db_path)
