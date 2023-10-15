__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import os
from .utils.logger import logger
import getpass
import datetime
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("fytok")
except PackageNotFoundError:
    __version__ = "unknown version"
logger.info(rf"""
#######################################################################################################################
    ______      _____     _
   / ____/_  __|_   _|__ | | __
  / /_  / / / /  | |/ _ \| |/ /
 / __/ / /_/ /   | | (_) |   <
/_/    \__, /    |_|\___/|_|\_\
      /____/      

 Copyright (c) 2021-present Zhi YU (Institute of Plasma Physics Chinese Academy of Sciences) 
 version = {__version__} URL: https://gitee.com/openfusion/fytok_tutorial

 Run by  {getpass.getuser().capitalize()} at {datetime.datetime.now().isoformat()}
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
