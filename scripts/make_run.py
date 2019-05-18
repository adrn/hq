# Standard library
import os
from os import path

# Project
import hq
from hq.config import HQ_CACHE_PATH
from hq.log import logger
from hq.script_helpers import get_parser


def make_run(name, allstar, allvisit):
    run_path = path.join(HQ_CACHE_PATH, name)

    if path.exists(run_path):
        logger.debug("Run already exists at '{}'! Delete that path and re-run "
                     "if you want to create a new run with this name."
                     .format(run_path))
        return

    logger.debug("Creating new run '{}' at path {}".format(name, run_path))
    os.makedirs(run_path)

    # Now copy in the template configuration file:
    HQ_ROOT_PATH = path.split(hq.__file__)[0]
    with open(path.join(HQ_ROOT_PATH, 'template_config.yml'), 'r') as f:
        template_config = f.read()

    with open(path.join(run_path, 'config.yml'), 'w') as f:
        f.write(template_config.format(
            run_name=name,
            allstar=path.abspath(path.expanduser(allstar)),
            allvisit=path.abspath(path.expanduser(allvisit))))


if __name__ == "__main__":
    # Define parser object
    parser = get_parser(description='Generate an HQ run template with the '
                                    'specified name. After running this, you '
                                    'then have to go in to the run directory '
                                    'and edit the configuration.',
                        loggers=logger)

    parser.add_argument("--name", dest="run_name", required=True,
                        type=str, help="The name of the run.")
    parser.add_argument("--allstar", dest="allstar", required=True,
                        type=str, help="Path to the allStar file.")
    parser.add_argument("--allvisit", dest="allvisit", required=True,
                        type=str, help="Path to the allVisit file.")

    args = parser.parse_args()

    make_run(args.run_name, args.allstar, args.allvisit)
