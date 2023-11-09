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
from spdm.data.Expression import Expression 
from spdm.data.Function import Function 
from spdm.data.HTree import Dict, HTree, List
from spdm.data.Signal import Signal, SignalND
from spdm.data.sp_property import SpTree, sp_property, sp_tree, AttributeTree
from spdm.data.TimeSeries import TimeSeriesAoS, TimeSlice
from spdm.geometry.Curve import Curve
from spdm.utils.tree_utils import merge_tree_recursive, update_tree
from spdm.utils.typing import array_type
from spdm.utils.tags import _not_found_

from ..utils.logger import logger
from ..utils.envs import FY_JOBID


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
    version: str
    copyright: str
    repository: str
    parameters: AttributeTree
    output_flag: array_type
    library: List[Library]

    def __str__(self) -> str:
        desc = {
            "name": self.name or self._parent.__class__.__name__,
            "version": self.version,
            "copyright": self.copyright,
        }

        return ", ".join(
            [
                f"{key}='{value}'"
                for key, value in desc.items()
                if value is not _not_found_ and value is not None and value != ""
            ]
        )


@sp_tree
class Identifier:
    name: str
    index: int
    description: str


@sp_tree
class Module(Actor):
    _plugin_prefix = __package__
    _plugin_registry = {}

    @classmethod
    def _plugin_guess_name(cls, self, cache, *args, **kwargs) -> str:
        pth = Path("code/name")
        plugin_name = (
            pth.fetch(cache, default_value=None)
            or pth.fetch(self.__class__._metadata, default_value=None)
            or pth.fetch(kwargs.get("default_value", _not_found_), default_value=None)
        )

        return plugin_name

    def __init__(self, *args, **kwargs):
        cache, entry, parent, kwargs = self.__class__._parser_args(*args, **kwargs)
        super().__init__(cache, _entry=entry, _parent=parent, **kwargs)

    @property
    def tag(self) -> str:
        return f"{FY_JOBID}/{self._plugin_prefix}{self.code.name or self.__class__.__name__.lower()}"

    code: Code


class IDS(Module):
    """Base class of IDS"""

    ids_properties: IDSProperties
    """Interface Data Structure properties. This element identifies the node above as an IDS"""


@sp_tree
class RZTuple:
    r: typing.Any
    z: typing.Any


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


@sp_tree(default_value=np.nan, force=True)
class CoreRadialGrid:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.psi_norm is _not_found_:
            if self.psi is _not_found_:
                raise ValueError("psi_norm or psi must be provided")
            elif self.psi_axis is _not_found_ or self.psi_boundary is _not_found_:
                self["psi_axis"] = self.psi.min()
                self["psi_boundary"] = self.psi.max()
            self["psi_norm"] = (self.psi - self.psi_axis) / (self.psi_boundary - self.psi_axis)
        elif self.psi is _not_found_ and not (self.psi_axis is _not_found_ or self.psi_boundary is _not_found_):
            self["psi"] = self.psi_norm * (self.psi_boundary - self.psi_axis) + self.psi_axis
        # elif self.psi_axis is _not_found_ or self.psi_boundary is _not_found_:
        #     self["psi_axis"] = self.psi.min()
        #     self["psi_boundary"] = self.psi.max()

        if self.rho_tor_norm is _not_found_:
            if self.rho_tor is _not_found_:
                raise ValueError("rho_tor_norm or rho_tor must be provided")
            elif self.rho_tor_boundary is _not_found_:
                self["rho_tor_boundary"] = self.rho_tor.max()
            self["rho_tor_norm"] = self.rho_tor / self.rho_tor_boundary
        elif self.rho_tor is _not_found_:
            if self.rho_tor_boundary is _not_found_:
                raise ValueError("rho_tor_boundary must be provided")
            self["rho_tor"] = self.rho_tor_norm * self.rho_tor_boundary
        elif self.rho_tor_boundary is _not_found_:
            self["rho_tor_boundary"] = self.rho_tor.max()

    def __copy__(self):
        return self.__class__(
            {
                "psi_axis": self.psi_axis,
                "psi_boundary": self.psi_boundary,
                "psi_norm": self.psi_norm,
                "rho_tor_boundary": self.rho_tor_boundary,
                "rho_tor_norm": self.rho_tor_norm,
            }
        )

    def remesh(self, rho_tor_norm=None, psi_norm=None) -> CoreRadialGrid:
        if rho_tor_norm is None or rho_tor_norm is _not_found_:
            if psi_norm is None or psi_norm is _not_found_:
                return self
            else:
                rho_tor_norm = Function(self.rho_tor_norm, self.psi_norm)(psi_norm)
        elif psi_norm is None or psi_norm is _not_found_:
            psi_norm = Function(self.psi_norm, self.rho_tor_norm)(rho_tor_norm)
        else:
            logger.warning("Both rho_tor_norm and psi_norm are provided! ")

        self.__init__(
            {
                "psi_axis": self.psi_axis,
                "psi_boundary": self.psi_boundary,
                "rho_tor_boundary": self.rho_tor_boundary,
                "psi_norm": psi_norm,
                "rho_tor_norm": rho_tor_norm,
            }
        )

        return self

    def duplicate(self, *args, **kwargs) -> CoreRadialGrid:
        g = self.__copy__()
        g.remesh(*args, **kwargs)
        return g

    psi_axis: float
    psi_boundary: float
    psi_norm: array_type
    psi: array_type

    rho_tor_boundary: float
    rho_tor_norm: array_type
    rho_tor: array_type

    @sp_property
    def rho_pol_norm(self) -> array_type:
        return np.sqrt(self.psi_norm)


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
