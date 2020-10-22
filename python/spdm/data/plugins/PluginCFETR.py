import os
import pathlib

from spdm.data.plugin.PluginTokamak import TokamakCollection
from spdm.util.logger import logger
from spdm.util.urilib import urisplit, uriunsplit

from ..Collection import Collection


class CFETRCollection(TokamakCollection):
    DEVICE_NAME = "CFETR"

    def __init__(self, uri, *args, mapping=None,   **kwargs):
        if isinstance(uri, str):
            uri = urisplit(uri)

        # path = getattr(uri, "path", None) or pathlib.Path.home()/f"public_data/~t/imas/3"

        # source = Collection(uriunsplit("mdsplus", uri.authority, path, None, uri.fragment), *args, **kwargs)

        super().__init__(uri, *args,  device_name=CFETRCollection.DEVICE_NAME, mapping=mapping, **kwargs)


__SP_EXPORT__ = CFETRCollection
