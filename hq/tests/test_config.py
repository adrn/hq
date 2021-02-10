# Third-party
import pathlib
import pytest
import yaml

# Project
from ..config import Config


@pytest.fixture(scope='module')
def make_config(tmpdir_factory):
    test_path = pathlib.Path(__file__).absolute().parent
    config_file = (test_path / '../pkgdata' / '_test_config.yml').resolve()
    new_config_file = tmpdir_factory.mktemp('config') / config_file.name
    allvisit_file = test_path / 'test-allVisit.fits'

    with open(config_file, 'r') as f:
        vals = yaml.safe_load(f.read())
    vals['input_data_file'] = str(allvisit_file)

    with open(new_config_file, 'w') as f:
        f.write(yaml.dump(vals))

    return new_config_file


def test_config(make_config):
    config_file = make_config

    c = Config(config_file)

    assert c.name == 'hqtest'
    assert c.input_data_file.exists()
    assert c.input_data_format is None

    assert c.source_id_colname == 'APOGEE_ID'
    assert c.max_prior_samples == 1_000_000
