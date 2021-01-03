import os
import pathlib
import collections
from spdm.util.logger import logger
from spdm.util.urilib import urisplit
from spdm.util.AttributeTree import AttributeTree

from ..Collection import Collection
from ..Node import Node

from spdm.data.db.Mapping import MappingCollection
from spdm.data.Document import Document
from spdm.data.File import File


class TokamakCollection(MappingCollection):

    def __init__(self, desc, *args, source=None, id_hasher=None, device_name="EAST", mapping=None, **kwargs):
        if isinstance(desc, str):
            desc = urisplit(desc)

        schema = desc.schema.split('+')

        self._device_name = (device_name or schema[0]).upper()

        schema = "+".join(schema[1:])

        if isinstance(mapping, Node):
            mapping_conf_files = None
            pass
        elif mapping is None:
            mapping_conf_files = []
        elif isinstance(mapping, str):
            mapping_conf_files = [mapping]
        elif isinstance(mapping, collections.abc.Sequence):
            mapping_conf_files = mapping
        else:
            raise TypeError(f"{type(mapping)}")

        if mapping_conf_files is not None:
            DEVICE_MAPPING_DIR = os.environ.get(
                "DEVICE_MAPPING_DIR", (pathlib.Path(__file__)/"../../../../../devices").resolve())

            SPDB_NAMELIST_VERSION = os.environ.get("SPDB_NAMELIST_VERSION",  "imas/3")

            mapping_conf_guess = ["config.xml", "static/config.xml", "dynamic/config.xml"]

            for file_name in mapping_conf_guess:
                p = pathlib.Path(f"{DEVICE_MAPPING_DIR}/{device_name}/{SPDB_NAMELIST_VERSION}/{file_name}")
                if p.exists():
                    mapping_conf_files.append(p)

            mapping = File(file_format="file/XML", path=mapping_conf_files)

        if source is None:
            path = desc.path or pathlib.Path.home() / f"public_data/{self._device_name}/{SPDB_NAMELIST_VERSION}/~t"

            target = Collection({
                "schema": schema,
                "authority": desc.authority,
                "path": path,
                "query": desc.query,
                "fragment": desc.fragment
            },  *args, **kwargs)

        super().__init__(desc, *args, target=target, mapping=mapping, id_hasher=id_hasher or "{shot}", **kwargs)


__SP_EXPORT__ = TokamakCollection
