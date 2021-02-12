# Project
from hq.config import Config
from hq.log import logger


def get_status(run_path):
    config_path = run_path / 'config.yml'
    if not config_path.exists():
        logger.error("Cannot get the status for this run: the config path "
                     f"does not exist! Expected path: {str(run_path)}")
        return None, None

    try:
        c = Config(config_path)
    except Exception:
        logger.error("Could not parse the config.yml file for this run! Are "
                     "you sure you edited the default values? Path: "
                     f"{str(config_path)}")
        return None, None

    status = {}

    status['make_prior_cache'] = c.prior_cache_file.exists()
    status['make_tasks'] = c.tasks_file.exists()
    status['run_thejoker'] = c.joker_results_file.exists()
    status['analyze_thejoker'] = c.metadata_joker_file.exists()
    status['run_mcmc'] = c.mcmc_results_file.exists()
    status['analyze_mcmc'] = c.metadata_mcmc_file.exists()
    status['run_constant'] = c.constant_results_file.exists()
    status['combine_metadata'] = c.metadata_file.exists()

    return c.name, status
