import logging
import os
import sys

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


__all__ = ["logger"]
