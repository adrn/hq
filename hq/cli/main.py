# Standard library
import argparse
import sys

# Package
from .helpers import get_parser


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
