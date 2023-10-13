import logging
import os
import sys
import atexit

from spdm.utils.logger import logger, sp_enable_logging

from .envs import *

if sys.version_info >= (3, 11):
    logger.replace(__package__[: __package__.find(".")])
else:
    logger = sp_enable_logging(
        __package__[: __package__.find(".")],
        level=FY_DEBUG,
        handler="STDOUT",
    )

def _at_end():
    logger.setLevel(logging.INFO)
    logger.info("The End")
    logging.shutdown()


atexit.register(_at_end)

__all__ = ["logger"]
