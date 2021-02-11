# Standard library
import argparse
import sys

# Package
from .helpers import get_parser
from .init_run import init_run

usage = '''
hq <command> [<args>]

Available commands:
    init        Initialize an HQ run
'''


class CLI:

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='A pipeline utility for running The Joker',
            usage=usage.strip())

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print(f"Unsupported command '{args.command}'")
            parser.print_help()
            sys.exit(1)

        getattr(self, args.command)()

    def init(self):
        parser = get_parser(
            description=(
                "Initialize an HQ run:"
                "Generate an HQ run template in the specified directory. "
                "If the path does not exist yet, it will be created. "
                "After running this, you should go into the run directory "
                "and edit the configuration file, config.yml."))
        parser.usage = 'hq init' + parser.format_usage()[9:]  # HACK
        args = parser.parse_args(sys.argv[2:])
        init_run(args.run_path)


def main():
    CLI()
