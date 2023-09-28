from __future__ import annotations

import functools
import typing
from dataclasses import dataclass
from enum import IntFlag

import numpy as np
from spdm.data.Path import Path
from spdm.data.Actor import Actor
from spdm.data.AoS import AoS
from spdm.data.Expression import Expression
from spdm.data.Field import Field
from spdm.data.Function import Function
from spdm.data.HTree import Dict, HTree, List
from spdm.data.Signal import Signal, SignalND
from spdm.data.sp_property import SpTree, sp_property
from spdm.data.TimeSeries import TimeSeriesAoS, TimeSlice
from spdm.geometry.Curve import Curve
from spdm.utils.tree_utils import merge_tree_recursive
from spdm.utils.typing import array_type
from spdm.utils.tags import _not_found_

from .._imas.lastest.utilities import _T_curved_object  # TODO: implement
from .._imas.lastest.utilities import _T_polarizer  # TODO: implement
from .._imas.lastest.utilities import (_T_core_radial_grid,
                                       _T_detector_aperture,
                                       _T_ids_properties, _T_library,
                                       _T_rz0d_dynamic_aos,
                                       _T_rz1d_dynamic_aos)

from .logger import logger


class Library(SpTree):
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

    parameters: Dict = sp_property(type="constant")
    """List of the code specific parameters in XML format"""


class Code(SpTree):
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

    parameters: Dict = sp_property(type="constant")
    """List of the code specific parameters in XML format"""

    output_flag: array_type = sp_property(coordinate1="/time", type="dynamic")
    """Output flag : 0 means the run is successful, other values mean some difficulty
       has been encountered, the exact meaning is then code specific. Negative values
       mean the result shall not be used."""

    library: List[Library] = sp_property(coordinate1="1...N")
    """List of external libraries used by the code that has produced this IDS"""


class Module(Actor):
    _plugin_registry = {}

    def __init__(self, *args, **kwargs):

        cache, entry, default_value, parent,  kwargs = HTree._parser_args(*args, **kwargs)

        if self.__class__ is Module or "_plugin_prefix" in vars(self.__class__):

            default_value = merge_tree_recursive(
                getattr(self.__class__, "_plugin_config", {}), default_value)

            pth = Path("code/name")

            plugin_name = pth.fetch(cache, default_value=None) or \
                pth.fetch(default_value, default_value=None) or \
                pth.fetch(kwargs, default_value=None)

            self.__class__.__dispatch_init__([plugin_name],
                                             self, cache, entry=entry, default_value=default_value,
                                             parent=parent, **kwargs)

            return

        super().__init__(cache, entry=entry,   default_value=default_value, parent=parent, **kwargs)

        logger.debug(f"""Load module {self.__class__.__name__}  
###############################################################################
{self.code.dump()}
{self.__class__.__doc__}
###############################################################################
""")

    code: Code = sp_property()
    """Generic decription of the code-specific parameters for the code that has produced this IDS"""


class IDS(Module):
    """Base class of IDS"""

    ids_properties: _T_ids_properties = sp_property()
    """Interface Data Structure properties. This element identifies the node above as an IDS"""

    time: array_type = sp_property(type="dynamic", units="s", ndims=1, data_type=float)
    """Generic time"""

    def advance(self, *args, time=None, **kwargs):
        if time is not None:
            self.time.append(time)
        super().advance(*args, time=time, **kwargs)

    def refresh(self, *args, **kwargs):
        super().refresh(*args, **kwargs)


RZTuple1D = _T_rz1d_dynamic_aos
RZTuple = _T_rz0d_dynamic_aos
# class RZTuple(_T_rz1d_dynamic_aos):
#     r = sp_property(type="dynamic", units="m", ndims=1, data_type=float)
#     z = sp_property(type="dynamic", units="m", ndims=1, data_type=float)
# CoreRadialGrid = _T_core_radial_grid


class CurveRZ(SpTree):

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1 and not isinstance(args[0], array_type):
            d = args[0]
            args = ()
        else:
            d = None

        super().__init__(d, **kwargs)

        if len(args) == 0:
            args = [np.vstack([super().get("r"), super().get("z")])]

        Curve.__init__(self, *args)

    @sp_property
    def r(self) -> array_type: return self.points[0]

    @sp_property
    def z(self) -> array_type: return self.points[1]


@dataclass
class RZTuple_:
    r: array_type | Expression
    z: array_type | Expression


class CoreRadialGrid(_T_core_radial_grid):
    """1D radial grid for core profiles"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,  **kwargs)
        rho_tor_norm = super().rho_tor_norm
        if rho_tor_norm is _not_found_:
            if self.rho_tor_boundary is not _not_found_:
                pass
            elif super().rho_tor.__value__ is _not_found_:
                raise RuntimeError(f"Can not find rho_tor_norm or rho_tor_boundary")
            else:
                self._cache["rho_tor_boundary"] = super().rho_tor.__value__[-1]

            self._cache["rho_tor_norm"] = super().rho_tor.__value__/self.rho_tor_boundary

    @sp_property
    def r0(self) -> float: return self.get("../../vacuum_toroidal_field/r0")

    @sp_property
    def b0(self) -> float:
        time = self.get("../time", 0.0)
        return self.get("../../vacuum_toroidal_field/b0")(time)

    def remesh(self, _rho_tor_norm: array_type) -> CoreRadialGrid:

        return CoreRadialGrid({

            "rho_tor_norm": _rho_tor_norm,

            "psi_norm": Function(self.psi_norm, self.rho_tor_norm)(_rho_tor_norm),

            "psi_magnetic_axis": self.psi_magnetic_axis,

            "psi_boundary": self.psi_boundary,

            "rho_tor_boundary": self.rho_tor_boundary,
        },
            parent=self._parent
        )

    psi_magnetic_axis: float = sp_property()

    psi_boundary: float = sp_property()

    rho_tor_boundary: float = sp_property()

    rho_tor_norm: array_type = sp_property(type="dynamic",  units="-")

    psi_norm: array_type = sp_property(type="dynamic", units="-")

    @sp_property
    def rho_pol_norm(self) -> array_type: return np.sqrt(self.psi_norm)

    @sp_property(type="dynamic", coordinate1="../rho_tor_norm", units="m")
    def rho_tor(self) -> Function: return self.rho_tor_norm*self.rho_tor_boundary

    @sp_property()
    def psi(self) -> Function:
        return self.psi_norm * (self.psi_boundary - self.psi_magnetic_axis) + self.psi_magnetic_axis


class DetectorAperture(_T_detector_aperture):
    def __geometry__(self, view="RZ", **kwargs):
        geo = {}
        styles = {}
        return geo, styles


__all__ = ["IDS", "Module", "Code", "Library",
           "DetectorAperture", "CoreRadialGrid", "RZTuple", "RZTuple1D", "CurveRZ",
           "array_type", "Function", "Field",
           "HTree", "List", "Dict", "SpTree", "sp_property",
           "AoS", "TimeSeriesAoS", "TimeSlice",
           "Signal", "SignalND",
           "IntFlag"]
