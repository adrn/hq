# Third-party
import pathlib
import pytest
import yaml


@pytest.fixture(scope='session')
def make_config(tmpdir_factory):
    test_path = pathlib.Path(__file__).absolute().parent
    config_file = (test_path / '../pkgdata' / '_test_config.yml').resolve()
    prior_file = (test_path / '../pkgdata' / '_test_prior.py').resolve()
    new_config_file = tmpdir_factory.mktemp('config') / config_file.name
    allvisit_file = test_path / 'test-allVisit.fits'

    with open(config_file, 'r') as f:
        vals = yaml.safe_load(f.read())
    vals['input_data_file'] = str(allvisit_file)
    vals['prior_file'] = str(prior_file)

    with open(new_config_file, 'w') as f:
        f.write(yaml.dump(vals))

    return new_config_file
