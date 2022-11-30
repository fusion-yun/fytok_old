import datetime
import getpass
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, TypeVar

from spdm.data import (Dict, File, Function, Link, List, Node, Path, Query,
                       sp_property)
from spdm.logger import logger
from spdm.tags import _undefined_

from .Module import Module


class IDSProperties(Dict):

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


class IDSCode(Dict):

    @sp_property
    def name(self) -> str:
        """Name of software generating IDS {constant}	STR_0D"""
        return self.get("name", None) or f"{self._parent.__class__.__module__}.{self._parent.__class__.__name__}"

    commit: str = sp_property()
    """	Unique commit reference of software {constant}	STR_0D"""

    version: str = sp_property()
    """Unique version (tag) of software {constant}	STR_0D"""

    repository: str = sp_property()
    """URL of software repository {constant}	STR_0D"""

    parameters: Dict = sp_property()
    """List of the code specific parameters  {constant}	dict"""

    output_flag: List[int] = sp_property()
    """Output flag : 0 means the run is successful, other values mean some difficulty has been encountered, 
           the exact meaning is then code specific. Negative values mean the result shall not be used. {dynamic}	INT_1D	1- time"""

    @dataclass
    class LibraryDesc:
        name: str = ""          # Name of software {constant}	STR_0D
        commit: str = ""        # Unique commit reference of software {constant}	STR_0D
        version: str = ""       # Unique version (tag) of software {constant}	STR_0D
        repository: str = ""    # URL of software repository {constant}	STR_0D
        parameters: list = None   # List of the code specific parameters in XML format {constant}

    library: List[LibraryDesc] = sp_property()
    "List of external libraries used by the code that has produced this IDS	struct_array [max_size=10]	1- 1...N"


class IDS(Module):
    """
        %%%DESCRIPTION%%%.
        .. todo:: '___NAME___' IS NOT IMPLEMENTED
    """
    _IDS = "NOT_DEFINED"

    def __serialize__(self, properties: Optional[Sequence] = None):
        res = super().__serialize__(properties=properties)
        res["@ids"] = self._IDS
        return res

    @classmethod
    def __deserialize__(cls, desc: Mapping):
        ids = desc.get("@ids", None)
        if ids is None:
            raise ValueError(desc)
        else:
            raise NotImplementedError(ids)

    ids_properties: IDSProperties = sp_property(default={})

    code: IDSCode = sp_property(default={})
