from argparse import ArgumentParser
import logging
import os

from astropy.utils.misc import isiterable


__all__ = ['get_parser']


def set_log_level(args, loggers):

    if args.verbosity == 1:
        log_level = logging.DEBUG

    elif args.verbosity == 2:
        log_level = 1

    elif args.verbosity == 3:
        log_level = 0

    elif args.quietness == 1:
        log_level = logging.WARNING

    elif args.quietness == 2:
        log_level = logging.ERROR

    else:
        log_level = logging.INFO  # default

    if not isiterable(loggers):
        loggers = [loggers]

    for logger in loggers:
        logger.setLevel(log_level)


def get_parser(description="", loggers=None):
    if loggers is None:
        loggers = []

    class CustomArgumentParser(ArgumentParser):

        def parse_args(self, *args, **kwargs):
            parsed = super().parse_args(*args, **kwargs)

            if parsed.run_name == '':
                raise ValueError("You must either specify a run name with the "
                                 "command line argument --name, or by setting "
                                 "the environment variable $HQ_RUN")

            set_log_level(parsed, loggers)
            return parsed

    # Define parser object
    parser = CustomArgumentParser(description=description)

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0,
                          dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0,
                          dest='quietness')

    parser.add_argument("--name", dest="run_name",
                        default=os.environ.get('HQ_RUN', ''),
                        type=str, help="The name of the run.")

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true",
                        help="Overwrite any existing results for this run.")

    return parser
