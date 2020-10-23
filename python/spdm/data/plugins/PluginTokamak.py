import os
import pathlib

from spdm.util.logger import logger
from spdm.util.urilib import urisplit, uriunsplit

from ..Collection import Collection
from spdm.data.plugins.PluginMapping import MappingCollection


class TokamakCollection(MappingCollection):

    def __init__(self, uri, *args, source=None, id_hasher=None, device_name="EAST", mapping=None, **kwargs):
        if isinstance(uri, str):
            uri = urisplit(uri)

        schema = uri.schema.split('+')

        self._device_name = (device_name or schema[0]).upper()

        schema = "+".join(schema[1:])

        if mapping is None:
            mapping = []

        DEVICE_MAPPING_DIR = os.environ.get(
            "DEVICE_MAPPING_DIR", (pathlib.Path(__file__)/"../../../../../devices").resolve())

        SPDB_NAMELIST_VERSION = os.environ.get("SPDB_NAMELIST_VERSION",  "imas/3")

        suffix_guess = ["config.xml", "static/config.xml", "dynamic/config.xml"]

        for suffix in suffix_guess:
            p = pathlib.Path(f"{DEVICE_MAPPING_DIR}/{device_name}/{SPDB_NAMELIST_VERSION}/{suffix}")
            if p.exists():
                mapping.append(p)

        if source is None:

            path = getattr(uri, "path", None) or pathlib.Path.home() / \
                f"public_data/{self._device_name}/{SPDB_NAMELIST_VERSION}/~t"

            source = Collection(uriunsplit(schema, uri.authority, path, None, uri.fragment),
                                *args, **kwargs)

        super().__init__(uri, *args, source=source, mapping=mapping, id_hasher=id_hasher or "{shot}", **kwargs)


__SP_EXPORT__ = TokamakCollection 
