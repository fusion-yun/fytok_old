__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import os
from pathlib import Path
from .utils.logger import logger
from .__version__ import __version__,__copyright__

logger.info(
    rf"""
#######################################################################################################################
    ______      _____     _
   / ____/_  __|_   _|__ | | __
  / /_  / / / /  | |/ _ \| |/ /
 / __/ / /_/ /   | | (_) |   <
/_/    \__, /    |_|\___/|_|\_\
      /____/      
{__copyright__}
version = {__version__}
#######################################################################################################################
"""
)


mapping_path = Path(__file__).parent.resolve() / "_mapping"

if mapping_path.exists():
    SP_DATA_MAPPING_PATH = (
        ":".join([mapping_path.as_posix(), os.environ.get("SP_DATA_MAPPING_PATH", "")])
    ).strip(":")

    os.environ["SP_DATA_MAPPING_PATH"] = SP_DATA_MAPPING_PATH

    logger.info(f"FyTok Mapping path: {SP_DATA_MAPPING_PATH}")
