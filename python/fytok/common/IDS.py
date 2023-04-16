import datetime
import getpass
import os
import typing
from dataclasses import dataclass
from spdm.data.Node import Node

from spdm.data.Dict import Dict
from spdm.data.List import List
from spdm.data.sp_property import sp_property

from .Module import Module


class IDSProperties(Dict[Node]):

    comment: str = sp_property()
    """ Any comment describing the content of this IDS {constant}	STR_0D"""

    @sp_property
    def homogeneous_time(self) -> float:
        h_time = getattr(self._parent.__class__, "_homogeneous_time", None)

        if h_time is None:
            pass
        elif hasattr(self._parent.__class__, "time_slice"):
            h_time = 1
        else:
            h_time = 0

        return h_time

    @sp_property
    def source(self):
        """Source of the data (any comment describing the origin of the data : code, path to diagnostic signals, processing method, ...) {constant}	STR_0D	"""
        return self["source"] or f"Create by '{self._parent.__class__.__module__}.{self._parent.__class__.__name__ }'"

    provider: str = sp_property(default=getpass.getuser().capitalize())
    """Name of the person in charge of producing this data {constant}	STR_0D	"""

    creation_date: str = sp_property(default=datetime.datetime.now().isoformat())
    """Date at which this data has been produced {constant}	STR_0D	"""

    @dataclass
    class VersionPut:
        data_dictionary: str = os.environ.get("IMAS_DD_VER", 3)
        access_layer: str = os.environ.get("IMAS_AL_VER", 4)
        access_layer_language: str = "N/A"

    version_put: VersionPut = sp_property()
    """Version of the access layer package used to PUT this IDS"""


class IDS(Module):
    """
        %%%DESCRIPTION%%%.
        .. todo:: '___NAME___' IS NOT IMPLEMENTED
    """
    _IDS = None

    def __serialize__(self, properties: typing.Optional[typing.Sequence] = None):
        res = super().__serialize__(properties=properties)
        res["@ids"] = self._IDS
        return res

    @classmethod
    def __deserialize__(cls, desc: typing.Mapping):
        ids = desc.get("@ids", None)
        if ids is None:
            raise ValueError(desc)
        else:
            raise NotImplementedError(ids)

    ids_properties: IDSProperties = sp_property(default={})
