from argparse import ArgumentParser
import logging

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
        log_level = logging.INFO # default

    if not isiterable(loggers):
        loggers = [loggers]

    for logger in loggers:
        logger.setLevel(log_level)


def get_parser(description="", loggers=None):
    if loggers is None:
        loggers = []

    class CustomArgumentParser(ArgumentParser):

        def parse_args(self):
            args = super().parse_args()
            set_log_level(args, loggers)
            return args

    # Define parser object
    parser = CustomArgumentParser(description=description)

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0,
                          dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0,
                          dest='quietness')

    return parser
