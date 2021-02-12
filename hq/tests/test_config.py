# Standar library
import os
import pickle

# Third-party
import pytest

# Project
from ..config import Config


@pytest.mark.usefixtures("make_config")
def test_config(make_config):
    c = Config(make_config)

    assert c.name == 'hqtest'
    assert c.input_data_file.exists()
    assert c.input_data_format is None

    assert c.source_id_colname == 'APOGEE_ID'
    assert c.max_prior_samples == 1_000_000


@pytest.mark.usefixtures("make_config")
def test_load_data(make_config):
    c = Config(make_config)
    data = c.data
    assert len(data) > 1


@pytest.mark.usefixtures("make_config")
def test_load_prior(make_config):
    c = Config(make_config)

    prior = c.get_prior()
    prior_samples = prior.sample(size=100)
    assert len(prior_samples['P']) == 100


@pytest.mark.usefixtures("make_config")
def test_load_source_data(make_config):
    c = Config(make_config)

    source_id = c.data[c.source_id_colname][0]
    rvdata = c.get_source_data(source_id)
    assert len(rvdata.rv) >= 3


@pytest.mark.usefixtures("make_config")
def test_pickle(tmpdir, make_config):
    c = Config(make_config)
    c.data

    pickle_file = tmpdir / 'config.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(c, f)

    pickle_file_compare = tmpdir / 'data.pkl'
    with open(pickle_file_compare, 'wb') as f:
        pickle.dump(c.data, f)

    # Make sure the data file or anything in the cache gets dropped
    assert os.path.getsize(pickle_file) < os.path.getsize(pickle_file_compare)
