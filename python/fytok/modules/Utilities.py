from __future__ import annotations

import functools
import typing
from dataclasses import dataclass
from enum import IntFlag
import numpy as np
from spdm.data.Path import Path
from spdm.data.Actor import Actor
from spdm.data.AoS import AoS
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

from ..utils.logger import logger


@sp_tree
class IDSProperties:
    comment: str
    homogeneous_time: int
    provider: str
    creation_date: str
    version_put: SpTree
    provenance: SpTree


@sp_tree
class Library:
    name: str
    commit: str
    version: str
    repository: str
    parameters: SpTree


@sp_tree
class Code:
    name: str
    commit: str
    version: str = "0.0.0"
    copyright: str = ""
    repository: str
    parameters: SpTree
    output_flag: array_type
    library: List[Library]


@sp_tree
class Identifier:
    name: str
    index: int
    description: str


class Module(Actor):
    _plugin_prefix = __package__
    _plugin_registry = {}

    def __init__(self, *args, **kwargs):

        cache, entry, parent,   kwargs = HTree._parser_args(*args, **kwargs)

        if self.__class__ is Module or "_plugin_prefix" in vars(self.__class__):

            pth = Path("code/name")

            plugin_name = pth.fetch(cache, default_value=None) or \
                pth.fetch(self.__class__._metadata, default_value=None) or \
                pth.fetch(self.__class__._metadata.get("default_value", {}), default_value=None) or \
                pth.fetch(kwargs, default_value=None)

            self.__class__.__dispatch_init__([plugin_name],
                                             self, cache,
                                             _entry=entry,
                                             _parent=parent,
                                             **kwargs)

            logger.info(
                f"Load module   \t:'{self.code.name or self.__class__.__name__}'  VERSION='{self.code.version}'  COPYRIGHT: {self.code.copyright}")

            return

        cache = merge_tree_recursive(self.__class__._metadata.get("default_value", {}), cache)
        cache = merge_tree_recursive(cache, {"code": self.__class__._metadata.get("code", {})})

        super().__init__(cache, _entry=entry, _parent=parent,  **kwargs)

    code: Code = sp_property()


_TSlice = typing.TypeVar("_TSlice")


@sp_tree
class TimeBasedActor(Module, typing.Generic[_TSlice]):

    TimeSlice = _TSlice

    time_slice: TimeSeriesAoS[_TSlice]

    @property
    def current(self) -> _TSlice: return self.time_slice.current

    def refresh(self, *args, **kwargs):
        """update the last time slice"""
        self.time_slice.refresh(*args, **kwargs)

    def advance(self, *args, **kwargs):
        """advance time_series to next slice"""
        self.time_slice.advance(*args, **kwargs)


class IDS(Module):
    """Base class of IDS"""

    ids_properties: IDSProperties
    """Interface Data Structure properties. This element identifies the node above as an IDS"""



@sp_tree
class PointRZ:  # utilities._T_rz0d_dynamic_aos
    r: float
    z: float


@sp_tree
class CurveRZ:  # utilities._T_rz1d_dynamic_aos
    r: array_type
    z: array_type


@sp_tree
class VacuumToroidalField:
    r0: float
    b0: float


class CoreRadialGrid(SpTree):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,  **kwargs)
        rho_tor_norm = super().get("rho_tor_norm", _not_found_) or self._metadata.get("../grid/rho_tor_norm", _not_found_)
        if rho_tor_norm is _not_found_:
            if self.rho_tor_boundary is not _not_found_:
                pass
            else:
                rho_tor = super().get("rho_tor", _not_found_)
                if rho_tor is _not_found_:
                    raise RuntimeError(f"Can not find rho_tor_norm or rho_tor_boundary")
                else:
                    self._cache["rho_tor_boundary"] = rho_tor[-1]

            self._cache["rho_tor_norm"] = rho_tor / self.rho_tor_boundary

    @sp_property
    def r0(self) -> float: return self.get("../vacuum_toroidal_field/r0")

    @sp_property
    def b0(self) -> float: return self.get("../vacuum_toroidal_field/b0")

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

    rho_tor_norm: array_type = sp_property(units="-")

    psi_norm: array_type = sp_property(units="-")

    @sp_property
    def rho_pol_norm(self) -> array_type: return np.sqrt(self.psi_norm)

    @sp_property
    def rho_tor(self) -> array_type: return self.rho_tor_norm*self.rho_tor_boundary

    @sp_property
    def psi(self) -> array_type:
        return self.psi_norm * (self.psi_boundary - self.psi_magnetic_axis) + self.psi_magnetic_axis


class DetectorAperture:  # (utilities._T_detector_aperture):
    def __geometry__(self, view="RZ", **kwargs):
        geo = {}
        styles = {}
        return geo, styles


@sp_tree
class PlasmaCompositionIonState:
    label: str
    z_min: float = sp_property(units="Elementary Charge Unit")
    z_max: float = sp_property(units="Elementary Charge Unit")
    electron_configuration: str
    vibrational_level: float = sp_property(units="Elementary Charge Unit")
    vibrational_mode: str


@sp_tree
class PlasmaCompositionSpecies:
    label: str
    a: float  # = sp_property(units="Atomic Mass Unit", )
    z_n: float  # = sp_property(units="Elementary Charge Unit", )


@sp_tree
class PlasmaCompositionNeutralElement(SpTree):
    a: float  # = sp_property(units="Atomic Mass Unit", )
    z_n: float  # = sp_property(units="Elementary Charge Unit", )
    atoms_n: int


@sp_tree
class PlasmaCompositionIons:
    label: str
    element: AoS[PlasmaCompositionNeutralElement]
    z_ion: float  # = sp_property( units="Elementary Charge Unit")
    state: PlasmaCompositionIonState


class PlasmaCompositionNeutralState:
    label: str
    electron_configuration: str
    vibrational_level: float  # = sp_property(units="Elementary Charge Unit")
    vibrational_mode: str
    neutral_type: str


class PlasmaCompositionNeutral:
    label: str
    element: AoS[PlasmaCompositionNeutralElement]
    state: PlasmaCompositionNeutralState


@sp_tree
class DistributionSpecies(SpTree):
    type: str
    ion: PlasmaCompositionIons
    neutral: PlasmaCompositionNeutral


# __all__ = ["IDS", "Module", "Code", "Library",
#            "DetectorAperture", "CoreRadialGrid", "PointRZ",   "CurveRZ",
#            "array_type", "Function", "Field",
#            "HTree", "List", "Dict", "SpTree", "sp_property",
#            "AoS", "TimeSeriesAoS", "TimeSlice",
#            "Signal", "SignalND", "Identifier"
#            "IntFlag"]
