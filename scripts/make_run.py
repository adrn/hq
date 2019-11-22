# Standard library
import os

# Project
import hq
from hq.config import HQ_CACHE_PATH
from hq.log import logger
from hq.script_helpers import get_parser


def make_run(name):
    run_path = os.path.join(HQ_CACHE_PATH, name)

    if os.path.exists(run_path):
        logger.debug("Run already exists at '{}'! Delete that path and re-run "
                     "if you want to create a new run with this name."
                     .format(run_path))
        return

    logger.debug("Creating new run '{}' at path {}".format(name, run_path))
    os.makedirs(run_path)

    # Now copy in the template configuration file:
    HQ_ROOT_PATH = os.path.split(hq.__file__)[0]
    tmpl = os.path.join(HQ_ROOT_PATH, 'pkgdata', 'template_config.py')
    with open(tmpl, 'r') as f:
        template_config = f.read()

    new_config_path = os.path.join(run_path, 'config.py')
    with open(new_config_path, 'w') as f:
        f.write(template_config)

    logger.info(f"Created an HQ run at: {run_path}\n\tNow edit the "
                f"configuration file at: {new_config_path}")


if __name__ == "__main__":
    # Define parser object
    parser = get_parser(description='Generate an HQ run template with the '
                                    'specified name. After running this, you '
                                    'then have to go in to the run directory '
                                    'and edit the configuration.',
                        loggers=logger)

    args = parser.parse_args()
    make_run(args.run_name)
