# Third-party
import pytest

# Project
from ..config import Config


@pytest.mark.usefixtures("make_config")
def test_config(make_config):
    config_file = make_config

    c = Config(config_file)

    assert c.name == 'hqtest'
    assert c.input_data_file.exists()
    assert c.input_data_format is None

    assert c.source_id_colname == 'APOGEE_ID'
    assert c.max_prior_samples == 1_000_000
