# Standard library
import argparse
import sys

# Third-party
import numpy as np
from threadpoolctl import threadpool_limits

# Package
from .helpers import get_parser
from ..log import logger


class CLI:
    """To add a new subcommand, just add a new classmethod and a docstring!"""
    _usage = None

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='A pipeline utility for running The Joker',
            usage=self._usage.strip())

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print(f"Unsupported command '{args.command}'")
            parser.print_help()
            sys.exit(1)

        getattr(self, args.command)()

    def init(self):
        """Initialize an HQ run"""
        from .init_run import init_run
        parser = get_parser(
            description=(
                "Generate an HQ run template in the specified directory. "
                "If the path does not exist yet, it will be created. "
                "After running this, you should go into the run directory "
                "and edit the configuration file, config.yml."))
        # HACK
        parser.usage = 'hq init' + parser.format_usage()[9:]
        args = parser.parse_args(sys.argv[2:])
        init_run(args.run_path)

    def make_prior_cache(self):
        """Create the cache file of prior samples"""
        from .make_prior_cache import make_prior_cache
        parser = get_parser(
            description=(
                "Generate an HDF5 cache file containing prior samples. The "
                "number of prior samples generated is specified in this run's "
                "config.yml file, and the prior is specified in prior.py"))
        # HACK
        parser.usage = 'hq make_prior_cache' + parser.format_usage()[9:]

        parser.add_argument("-s", "--seed", dest="seed", default=None,
                            type=int, help="Random number seed")
        args = parser.parse_args(sys.argv[2:])

        with args.Pool(**args.Pool_kwargs) as pool:
            make_prior_cache(args.run_path, pool=pool,
                             overwrite=args.overwrite, seed=args.seed)

    def make_tasks(self):
        """Generate the data task file, used by MPI scripts."""
        from .make_tasks import make_tasks
        parser = get_parser(
            description=(
                "Generate an HDF5 file containing the data, but reprocessed "
                "into a format that is faster to read with MPI."),
            multiproc_options=False)
        # HACK
        parser.usage = 'hq make_tasks' + parser.format_usage()[9:]

        args = parser.parse_args(sys.argv[2:])

        with args.Pool() as pool:
            make_tasks(args.run_path, pool=pool, overwrite=args.overwrite)

    def run_thejoker(self):
        """Run The Joker on the input data"""
        from thejoker.logging import logger as joker_logger
        from .run_thejoker import run_thejoker

        parser = get_parser(
            description=(
                "This command is the main workhorse for HQ: it runs The Joker "
                "on all of the input data and caches the samplings."),
            loggers=[logger, joker_logger])
        # HACK
        parser.usage = 'hq run_thejoker' + parser.format_usage()[9:]

        parser.add_argument("-s", "--seed", dest="seed", default=None,
                            type=int, help="Random number seed")
        parser.add_argument("--limit", dest="limit", default=None,
                            type=int, help="Maximum number of stars to process"
                            )

        args = parser.parse_args(sys.argv[2:])

        if args.seed is None:
            args.seed = np.random.randint(2**32 - 1)
            logger.log(
                1, f"No random seed specified, so using seed: {args.seed}")

        with threadpool_limits(limits=1, user_api='blas'):
            with args.Pool(**args.Pool_kwargs) as pool:
                run_thejoker(run_path=args.run_path, pool=pool,
                             overwrite=args.overwrite,
                             seed=args.seed, limit=args.limit)

        sys.exit(0)

    def run_constant(self):
        """Fit a robust constant RV model to all source data"""
        from .run_constant import run_constant
        from threadpoolctl import threadpool_limits

        parser = get_parser(
            description=(
                "Fit a constant RV and a robust constant RV model to the data "
                "and record the log likelihoods. This can be used to compare "
                "to the likelihoods computed for the Keplerian orbit models."))
        # HACK
        parser.usage = 'hq run_constant' + parser.format_usage()[9:]

        args = parser.parse_args(sys.argv[2:])

        with threadpool_limits(limits=1, user_api='blas'):
            with args.Pool(**args.Pool_kwargs) as pool:
                run_constant(args.run_path, pool=pool,
                             overwrite=args.overwrite)

        sys.exit(0)

    def analyze_thejoker(self):
        """Analyze the samplings generated by The Joker"""
        from .analyze_joker_samplings import analyze_joker_samplings
        parser = get_parser(
            description=(
                "Generate statistics and metadata for the samplings "
                "created by running The Joker."))
        # HACK
        parser.usage = 'hq analyze_thejoker' + parser.format_usage()[9:]

        args = parser.parse_args(sys.argv[2:])

        with threadpool_limits(limits=1, user_api='blas'):
            with args.Pool(**args.Pool_kwargs) as pool:
                analyze_joker_samplings(args.run_path, pool=pool)


# Auto-generate the usage block:
cmds = []
maxlen = max([len(name) for name in CLI.__dict__.keys()])
for name, attr in CLI.__dict__.items():
    if not name.startswith('_'):
        cmds.append(f'    {name.ljust(maxlen)}  {attr.__doc__}\n')

CLI._usage = f"""
hq <command> [<args>]

Available commands:
{''.join(cmds)}
"""


def main():
    CLI()
