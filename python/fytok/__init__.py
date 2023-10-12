__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import os
from .utils.logger import logger
from .__version__ import __version__
from .__doc__ import copyright
logger.info(
    rf"""
#######################################################################################################################
{copyright}
version = {__version__}
#######################################################################################################################
"""
)

try:
    from importlib import resources as impresources
    from . import _mapping
    from spdm.data.Entry import EntryProxy

    EntryProxy._mapping_path.extend([p.resolve() for p in impresources.files(_mapping)._paths])

except Exception as error:
    raise FileNotFoundError(f"Can not find mappings!") from error
else:
    logger.info(f"Mapping path    \t: {EntryProxy._mapping_path}")
