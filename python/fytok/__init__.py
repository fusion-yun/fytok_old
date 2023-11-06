__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import os
try:
    for k, v in os.environ.items():
        if k.startswith("FY_"):
            os.environ[f"SP_{k[3:]}"] = v

    os.environ["SP_LABEL"] = "fytok"
except Exception:
    pass

from .ontology import GLOBAL_ONTOLOGY
from .__version__ import __version__
from .utils.logger import logger
import datetime
import getpass


############################################################


try:
    from .extension import tags as extension_tags
except ImportError:
    extension_tags = ""

try:
    from importlib import resources as impresources
    from . import _mapping
    from spdm.data.Entry import EntryProxy

    EntryProxy._mapping_path.extend([p.resolve() for p in impresources.files(_mapping)._paths])

except Exception as error:
    raise FileNotFoundError(f"Can not find mappings!") from error


logger.info(rf"""
#######################################################################################################################
    ______      _____     _
   / ____/_  __|_   _|__ | | __
  / /_  / / / /  | |/ _ \| |/ /
 / __/ / /_/ /   | | (_) |   <
/_/    \__, /    |_|\___/|_|\_\
      /____/

 Copyright (c) 2021-present Zhi YU (Institute of Plasma Physics Chinese Academy of Sciences)
            
 url: https://gitee.com/openfusion/fytok_tutorial 
      https://github.com/fusion-yun/fytok_tutorial

 version = {__version__} {extension_tags} 

 Run by {getpass.getuser()} on {os.uname().nodename} at {datetime.datetime.now().isoformat()}

#######################################################################################################################
""")
