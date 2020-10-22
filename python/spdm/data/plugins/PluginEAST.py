import os
import pathlib

from spdm.util.logger import logger
from spdm.util.urilib import urisplit, uriunsplit

from ..Collection import Collection
# from spdm.data.plugins.PluginMapping import MappingCollection
from spdm.data.plugins.PluginTokamak import TokamakCollection


class EASTCollection(TokamakCollection):
    DEVICE_NAME = "EAST"

    def __init__(self, uri, *args,      **kwargs):
        # if isinstance(uri, str):
        #     uri = urisplit(uri)
        # if mapping is None:
        #     mapping = []

        # EAST_MAPPING_DIR = os.environ.get(
        #     "EAST_MAPPING_DIR",
        #     (pathlib.Path(__file__)/"../../../../../mapping/EAST").resolve()
        # )

        # mapping.extend([f"{EAST_MAPPING_DIR}/imas/3/static/config.xml",
        #                 f"{EAST_MAPPING_DIR}/imas/3/dynamic/config.xml"])

        super().__init__(uri, *args, **kwargs)

    @property
    def device(self):
        return super().device() or EASTCollection.DEVICE_NAME


__SP_EXPORT__ = EASTCollection
