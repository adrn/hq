# Standard library
from os import path

# Third-party
import astropy.units as u
import yaml
from thejoker import TheJoker

# Project
from hq.config import (HQ_CACHE_PATH, config_to_jokerparams,
                       config_to_prior_cache)
from hq.log import logger
from hq.script_helpers import get_parser
from hq.sample_prior import make_prior_cache


def main(name, overwrite):
    run_path = path.join(HQ_CACHE_PATH, name)

    with open(path.join(run_path, 'config.yml'), 'r') as f:
        config = yaml.load(f.read())

    params = config_to_jokerparams(config)
    prior_cache_path = config_to_prior_cache(confi, params)

    if path.exists(prior_cache_path) and not overwrite:
        logger.debug("Prior cache file already exists at '{}'! User -o or "
                     "--overwrite to re-generate.".format(prior_cache_path))
        return

    n_samples = config['prior']['n_samples']
    logger.debug("Prior samples file not found - generating {} samples in "
                 "cache file at {}...".format(n_samples, prior_cache_path))
    make_prior_cache(prior_cache_path, TheJoker(params), nsamples=n_samples)
    logger.debug("...done generating cache.")


if __name__ == "__main__":
    # Define parser object
    parser = get_parser(description='Generate an HQ run template with the '
                                    'specified name. After running this, you '
                                    'then have to go in to the run directory '
                                    'and edit the configuration.',
                        loggers=logger)

    parser.add_argument("--name", dest="run_name", required=True,
                        type=str, help="The name of the run.")
    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true",
                        help="Overwrite any existing results for this run.")

    args = parser.parse_args()

    main(args.run_name, args.overwrite)
