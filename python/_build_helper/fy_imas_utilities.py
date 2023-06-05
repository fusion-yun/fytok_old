

import collections.abc

from spdm.utils.Pluggable import Pluggable
from spdm.data.NamedDict import NamedDict
import numpy as np


class _T_Library(SpDict):
    """
    Library used by the code that has produced this IDS
    """

    name: str = sp_property(type="constant")
    """Name of software"""

    commit: str = sp_property(type="constant")
    """Unique commit reference of software"""

    version: str = sp_property(type="constant")
    """Unique version (tag) of software"""

    repository: str = sp_property(type="constant")
    """URL of software repository"""

    parameters: NamedDict = sp_property(type="constant")
    """List of the code specific parameters in XML format"""


class _T_Code(SpDict):
    """
    Generic decription of the code-specific parameters for the code that has
       produced this IDS
    """

    name: str = sp_property(type="constant")
    """Name of software generating IDS"""

    commit: str = sp_property(type="constant")
    """Unique commit reference of software"""

    version: str = sp_property(type="constant")
    """Unique version (tag) of software"""

    repository: str = sp_property(type="constant")
    """URL of software repository"""

    parameters: NamedDict = sp_property(type="constant")
    """List of the code specific parameters in XML format"""

    output_flag: np.ndarray = sp_property(coordinate1="/time", type="dynamic")
    """Output flag : 0 means the run is successful, other values mean some difficulty
       has been encountered, the exact meaning is then code specific. Negative values
       mean the result shall not be used."""

    library: List[_T_Library] = sp_property(coordinate1="1...N")
    """List of external libraries used by the code that has produced this IDS"""


class _T_Module(SpDict, Pluggable):
    _plugin_registry = {}

    @classmethod
    def __dispatch__init__(cls, name_list, self,  *args, **kwargs) -> None:
        if name_list is None:
            module_name = None
            code = kwargs.get("code", None)
            if isinstance(code, collections.abc.Mapping):
                module_name = code.get("name", None)

            if len(args) == 0:
                pass
            elif module_name is None and isinstance(args[0], collections.abc.Mapping):
                module_name = args[0].get("code", {}).get("name", None)
            elif hasattr(args[0], "__as_entry__"):
                module_name = args[0].__as_entry__().get("code/name", None)

            if module_name is not None:
                prefix: str = getattr(self.__class__, "_IDS", self.__class__.__name__.lower())
                if prefix.startswith('_t_'):
                    prefix = prefix[3:]
                name_list = [f"fytok/plugins/{prefix}/{module_name}"]
        if name_list is None or len(name_list) == 0:
            return super().__init__(self, *args, **kwargs)
        else:
            return super().__dispatch__init__(name_list, self, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        if self.__class__ is _T_Module or "_plugin_registry" in vars(self.__class__):
            _T_Module.__dispatch__init__(None, self, *args, **kwargs)
            return
        super().__init__(*args, **kwargs)

    code: _T_Code = sp_property()
    """Generic decription of the code-specific parameters for the code that has produced this IDS"""


class _T_IDS(_T_Module):
    """ Base class of IDS """

    _plugin_registry = {}

    ids_properties: _T_ids_properties = sp_property()
    """Interface Data Structure properties. This element identifies the node above as an IDS"""

    time: np.ndarray = sp_property(type="dynamic", units="s", ndims=1, data_type=float)
    """Generic time"""

    def advance(self,  *args, time=None, ** kwargs):
        if time is not None:
            self.time.append(time)

    def update(self,  *args,  ** kwargs):
        super().update(*args, **kwargs)

# from spdm.geometry.Point import Point
# from spdm.geometry.CubicSplineCurve import CubicSplineCurve

# class PointRZ(Point):
#     @property
#     def r(self) -> float: return self.points[0]
#     @property
#     def z(self) -> float: return self.points[1]


# class CurveRZ(CubicSplineCurve):
#     @property
#     def r(self) -> np.ndarray: return self.points[0]

#     @property
#     def z(self) -> np.ndarray: return self.points[1]
