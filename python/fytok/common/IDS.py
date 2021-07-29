import datetime
import getpass
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, TypeVar

from spdm.data.Node import Dict, List, Node, sp_property
from spdm.util.logger import logger
from spdm.util.utilities import _undefined_
from spdm.flow.Actor import Actor


class IDSProperties(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,  **kwargs)

    @sp_property
    def comment(self):
        """ Any comment describing the content of this IDS {constant}	STR_0D"""
        return self["comment"]

    @sp_property
    def homogeneous_time(self):
        """This node must be filled (with 0, 1, or 2) for the IDS to be valid. If 1, the time of this IDS is homogeneous,
            i.e. If 1, the time values for this IDS are stored in the time node just below the root of this IDS. 
                 If 0, the time values are stored in the various time fields at lower levels in the tree.
                 In the case only constant or static nodes are filled within the IDS, homogeneous_time must be set to 2 {constant}	INT_0D	"""

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

    @sp_property
    def provider(self):
        """Name of the person in charge of producing this data {constant}	STR_0D	"""
        return self["provider"] or getpass.getuser().capitalize()

    @sp_property
    def creation_date(self):
        """Date at which this data has been produced {constant}	STR_0D	"""
        return self["creation_date"] or datetime.datetime.now().isoformat()

    @dataclass
    class VersionPut:
        data_dictionary: str = os.environ.get("IMAS_DD_VER", 3)
        access_layer: str = os.environ.get("IMAS_AL_VER", 4)
        access_layer_language: str = "N/A"

    @sp_property
    def version_put(self) -> VersionPut:
        """Version of the access layer package used to PUT this IDS"""
        return IDSProperties.VersionPut(**(self["version_put"]._as_dict()))


class IDSCode(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def name(self) -> str:
        """Name of software generating IDS {constant}	STR_0D"""
        return self.get("name", None) or f"{self._parent.__class__.__module__}.{self._parent.__class__.__name__}"

    @sp_property
    def commit(self) -> str:
        """	Unique commit reference of software {constant}	STR_0D"""
        return self["commit"]

    @sp_property
    def version(self) -> str:
        """Unique version (tag) of software {constant}	STR_0D"""
        return self["version"]

    @sp_property
    def repository(self) -> str:
        """URL of software repository {constant}	STR_0D"""
        return self["repository"]

    @sp_property
    def parameters(self) -> dict:
        r"""List of the code specific parameters  {constant}	dict"""
        return self.get("parameters", {})

    @sp_property
    def output_flag(self) -> Sequence[int]:
        """Output flag : 0 means the run is successful, other values mean some difficulty has been encountered, 
           the exact meaning is then code specific. Negative values mean the result shall not be used. {dynamic}	INT_1D	1- time"""
        return self["output_flag"]

    @dataclass
    class LibraryDesc:
        name: str = ""          # Name of software {constant}	STR_0D
        commit: str = ""        # Unique commit reference of software {constant}	STR_0D
        version: str = ""       # Unique version (tag) of software {constant}	STR_0D
        repository: str = ""    # URL of software repository {constant}	STR_0D
        parameters: list = None   # List of the code specific parameters in XML format {constant}

    @sp_property
    def library(self) -> List[LibraryDesc]:
        "List of external libraries used by the code that has produced this IDS	struct_array [max_size=10]	1- 1...N"
        return List[IDSCode.LibraryDesc](self["library"],   parent=self)


_TState = TypeVar("_TState")


class IDS(Actor[_TState]):
    """
        %%%DESCRIPTION%%%.
        .. todo:: '___NAME___' IS NOT IMPLEMENTED
    """
    _IDS = "NOT_DEFINED"

    def __init__(self,  *args, ** kwargs):
        super().__init__(*args, ** kwargs)

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

    @sp_property
    def ids_properties(self) -> IDSProperties:
        return self.get("ids_properties", {})

    @sp_property
    def code(self) -> IDSCode:
        return self.get('code', {})
