import os
import pathlib

from spdm.util.logger import logger
from spdm.util.urilib import urisplit, uriunsplit

from ..Collection import Collection
from  spdm.data.plugins.PluginMapping import MappingCollection


class TokamakCollection(MappingCollection):

    def __init__(self, *args, device_name="EAST", mapping=None,**kwargs):
        self._device_name = device_name

        if mapping is None:
            mapping = []

        DEVICE_MAPPING_DIR = os.environ.get("DEVICE_MAPPING_DIR", (pathlib.Path(__file__)/"../../../../../devices").resolve())

        SPDB_NAMELIST_VERSION = os.environ.get("SPDB_NAMELIST_VERSION",  "imas/3")

        suffix_guess = ["config.xml", "static/config.xml", "dynamic/config.xml"]

        for suffix in suffix_guess:
            p = pathlib.Path(f"{DEVICE_MAPPING_DIR}/{device_name}/{SPDB_NAMELIST_VERSION}/{suffix}")
            if p.exists():
                mapping.append(p)

        super().__init__(*args, mapping=mapping, **kwargs)


__SP_EXPORT__ = TokamakCollection
