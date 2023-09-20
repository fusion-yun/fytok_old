import logging
import os
import sys

from spdm.utils.logger import logger, sp_enable_logging

if sys.version_info >= (3, 11):
    logger.replace(__package__[:__package__.find('.')])
else:
    logger = sp_enable_logging(__package__[:__package__.find('.')],
                               level=os.environ.get("SP_DEBUG", "debug"),
                               handler="STDOUT")


FY_DEBUG = logger.level

__all__ = ["logger", "FY_DEBUG"]
