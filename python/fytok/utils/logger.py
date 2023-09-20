import os
import logging
from spdm.utils.logger import sp_enable_logging

logger = sp_enable_logging(__package__[:__package__.find('.')], handler="STDOUT")

FY_DEBUG: int = os.environ.get("FY_DEBUG", "2")

match FY_DEBUG:
    case "-1" | "quiet":
        logger.setLevel(logging.CRITICAL)
        FY_DEBUG = -1

    case "0" | "warning":
        logger.setLevel(logging.WARNING)
        FY_DEBUG = 0

    case "1" | "True" | "true" | "verbose" | "debug":
        logger.setLevel(logging.DEBUG)
        FY_DEBUG = 1
        
    case _:
        FY_DEBUG = -2
        logger.setLevel(logging.INFO)

__all__ = ["logger", "FY_DEBUG"]
