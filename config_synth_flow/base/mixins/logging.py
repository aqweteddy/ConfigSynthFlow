"""
Mixin class for logging functionality.
"""

import logging
from logging import Formatter, StreamHandler, getLogger


class LoggingMixin:
    """Mixin for logging functionality"""

    @property
    def logger(self):
        """
        Get the logger for the pipeline.

        Returns:
            logging.Logger: Logger for the pipeline.
        """
        if not hasattr(self, "_logger"):
            self._logger = getLogger(self.class_name)
            self._logger.setLevel(logging.INFO)
            handler = StreamHandler()
            formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s | %(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        return self._logger
