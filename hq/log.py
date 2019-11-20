# Standard library
import logging

# Third-party
from astropy.logger import StreamHandler

__all__ = ['logger']


class HQHandler(StreamHandler):
    def emit(self, record):
        record.origin = 'hq'
        super().emit(record)


class HQLogger(logging.getLoggerClass()):
    def _set_defaults(self):
        """Reset logger to its initial state"""

        # Remove all previous handlers
        for handler in self.handlers[:]:
            self.removeHandler(handler)

        # Set default level
        self.setLevel(logging.INFO)

        # Set up the stdout handler
        sh = HQHandler()
        self.addHandler(sh)


logging.setLoggerClass(HQLogger)
logger = logging.getLogger('hq')
logger._set_defaults()
