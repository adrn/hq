# Standard library
import pathlib

# Project
import hq
from hq.log import logger
from hq.script_helpers import get_parser


def init_run(run_path):
    new_config_path = run_path / 'config.yml'
    if new_config_path.exists():
        logger.info(f"Run already exists in '{new_config_path.parent}'! "
                    "Delete that path and re-run this script if you really "
                    "want to create a new run in that path.")
        return

    logger.debug(f"Creating a new HQ run in path {run_path}")
    run_path.mkdir(exist_ok=True)

    # Now copy in the template configuration file:
    HQ_ROOT_PATH = pathlib.Path(hq.__file__).parent
    tmpl = HQ_ROOT_PATH / 'pkgdata' / 'template_config.yml'
    with open(tmpl, 'r') as f:
        template_config = f.read()

    with open(new_config_path, 'w') as f:
        f.write(template_config)

    # Now copy template prior:
    prior_path = HQ_ROOT_PATH / 'pkgdata' / 'template_prior.py'
    new_prior_path = run_path / 'prior.py'
    with open(prior_path, 'r') as f:
        with open(new_prior_path, 'w') as new_f:
            new_f.write(f.read())

    logger.info(f"Created an HQ run at: {run_path}\n"
                f"\tNow edit the configuration file at: {new_config_path}\n"
                f"\tAnd edit the prior specification file at: {new_prior_path}"
                )


if __name__ == "__main__":
    # Define parser object
    parser = get_parser(
        description='Generate an HQ run template in the specified directory. '
                    'If the path does not exist yet, it will be created. '
                    'After running this, you should go into the run directory '
                    'and edit the configuration file, config.yml',
        loggers=logger)

    args = parser.parse_args()
    init_run(pathlib.Path(args.run_path))
