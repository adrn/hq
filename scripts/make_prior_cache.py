# Standard library
from os import path

# Project
from hq.config import Config
from hq.log import logger
from hq.script_helpers import get_parser
from hq.sample_prior import make_prior_cache


def main(name, overwrite, batch_size):
    c = Config.from_run_name(name)

    if path.exists(c.prior_cache_file) and not overwrite:
        logger.debug(f"Prior cache file already exists at "
                     f"{c.prior_cache_file}! Use -o / --overwrite "
                     f"to re-generate.")
        return

    logger.debug("Prior samples file not found: generating {c.n_prior_samples} "
                 "samples in cache file at {c.prior_cache_file}")
    make_prior_cache(c.prior_cache_file, c.prior, c.n_prior_samples,
                     batch_size=batch_size)
    logger.debug("...done generating cache.")


if __name__ == "__main__":
    # Define parser object
    parser = get_parser(description='Generate an HQ run template with the '
                                    'specified name. After running this, you '
                                    'then have to go in to the run directory '
                                    'and edit the configuration.',
                        loggers=logger)

    parser.add_argument("--batch_size", dest="batch_size", type=int,
                        default=None,
                        help="The number of samples to generate in each batch.")

    args = parser.parse_args()

    main(args.run_name, args.overwrite, args.batch_size)
