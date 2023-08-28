

import collections.abc
from enum import IntFlag

import numpy as np
from spdm.data.Actor import Actor
from spdm.data.AoS import AoS
from spdm.data.Field import Field
from spdm.data.Function import Function
from spdm.data.HTree import List
from spdm.data.NamedDict import NamedDict
from spdm.data.Signal import Signal, SignalND
from spdm.data.sp_property import SpDict, sp_property
from spdm.data.TimeSeries import TimeSeriesAoS, TimeSlice
from spdm.utils.plugin import Pluggable


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


class _T_Module(Actor):
    _plugin_registry = {}

    _plugin_prefix = "fytok/plugins/"

    _plugin_name_path = "code/name"

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
        super().advance(*args, time=time, **kwargs)

    def refresh(self,  *args,  ** kwargs):
        super().refresh(*args, **kwargs)

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
