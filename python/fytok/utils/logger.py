import logging
import os

from spdm.utils.logger import logger
__all__ = ["logger"]

from spdm.utils.logger import sp_enable_logging


logger = sp_enable_logging(__package__[:__package__.find('.')], handler="STDOUT")

FY_DEBUG = os.environ.get("FY_DEBUG", "2")

match FY_DEBUG:
    case "0" | "warning":
        logger.setLevel(logging.WARNING)
    case "2" | "True" | "true" | "verbose" | "debug":
        logger.setLevel(logging.DEBUG)
    case "-1" | "quiet":
        logger.setLevel(logging.CRITICAL)
    case _:
        logger.setLevel(logging.INFO)
