__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import os
from pathlib import Path
from .utils.logger import logger
from .__version__ import __version__, __copyright__

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

try:
    from importlib import resources as impresources
    from . import mapping
    from spdm.data.Entry import EntryProxy

    EntryProxy._mapping_path.extend(impresources.files(mapping)._paths)

except Exception as error:
    raise FileNotFoundError(f"Can not find mappings!") from error
else:
    logger.info(f"Mapping path {EntryProxy._mapping_path}")
