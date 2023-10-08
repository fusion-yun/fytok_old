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
from spdm.data.sp_property import SpTree, sp_property, sp_tree
from spdm.data.TimeSeries import TimeSeriesAoS, TimeSlice
from spdm.geometry.Curve import Curve
from spdm.utils.tree_utils import merge_tree_recursive
from spdm.utils.typing import array_type
from spdm.utils.tags import _not_found_

from .logger import logger


@sp_tree
class IDSProperties:
    """Interface Data Structure properties. This element identifies the node above as
        an IDS"""

    comment: str
    """ Any comment describing the content of this IDS"""

    homogeneous_time: int

    provider: str
    """ Name of the person in charge of producing this data"""

    creation_date: str
    """ Date at which this data has been produced"""

    version_put: SpTree = sp_property()
    """ Version of the access layer package used to PUT this IDS"""

    provenance: SpTree
    """ Provenance information about this IDS"""


class Library(SpTree):
    name: str  # = sp_property()
    commit: str  # = sp_property()
    version: str  # = sp_property()
    repository: str  # = sp_property()
    parameters: SpTree  # = sp_property()


@sp_tree
class Code:
    name: str               # = sp_property()
    commit: str             # = sp_property()
    version: str            # = sp_property()
    repository: str         # = sp_property()
    parameters: SpTree      # = sp_property()
    output_flag: array_type  # = sp_property()
    library: List[Library]  # = sp_property()


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

        if self.__class__.__doc__ is not None and self.code.version is not _not_found_:

            logger.info(f"""
###############################################################################
Load module {self.code.name or self.__class__.__name__}  version={self.code.version}
{self.__class__.__doc__}
###############################################################################
""")
        else:
            logger.info(f"""Load module {self.code.name or self.__class__.__name__} """)

    code: Code = sp_property()


class IDS(Module):
    """Base class of IDS"""

    ids_properties: ids_properties = sp_property()
    """Interface Data Structure properties. This element identifies the node above as an IDS"""

    time: array_type = sp_property(type="dynamic", units="s", ndims=1, data_type=float)
    """Generic time"""

    def advance(self, *args, time=None, **kwargs):
        if time is not None:
            self.time.append(time)
        super().advance(*args, time=time, **kwargs)

    def refresh(self, *args, **kwargs):
        super().refresh(*args, **kwargs)


@sp_tree
class PointRZ:  # utilities._T_rz0d_dynamic_aos
    r: float
    z: float


@sp_tree
class CurveRZ:  # utilities._T_rz1d_dynamic_aos
    r: array_type
    z: array_type


class CoreRadialGrid:  # (utilities._T_core_radial_grid):
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


class DetectorAperture:  # (utilities._T_detector_aperture):
    def __geometry__(self, view="RZ", **kwargs):
        geo = {}
        styles = {}
        return geo, styles


class Identifier(SpTree):
    name: str = sp_property(type="dynamic")
    index: int = sp_property(type="dynamic")
    description: str = sp_property(type="dynamic")


# __all__ = ["IDS", "Module", "Code", "Library",
#            "DetectorAperture", "CoreRadialGrid", "PointRZ",   "CurveRZ",
#            "array_type", "Function", "Field",
#            "HTree", "List", "Dict", "SpTree", "sp_property",
#            "AoS", "TimeSeriesAoS", "TimeSlice",
#            "Signal", "SignalND", "Identifier"
#            "IntFlag"]
