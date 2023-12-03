from __future__ import annotations

import functools
import typing
from dataclasses import dataclass
from enum import IntFlag
import numpy as np
from spdm.data.Path import Path, update_tree
from spdm.data.Actor import Actor
from spdm.data.AoS import AoS
from spdm.data.Field import Field
from spdm.data.Expression import Expression
from spdm.data.Function import Function
from spdm.data.HTree import Dict, HTree, List
from spdm.data.Signal import Signal, SignalND

from spdm.data.sp_property import SpTree, sp_property, sp_tree, PropertyTree
from spdm.data.TimeSeries import TimeSeriesAoS, TimeSlice
from spdm.geometry.Curve import Curve
from spdm.utils.typing import array_type, is_array
from spdm.utils.tags import _not_found_
from spdm.view import View as sp_view
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
    """代码名称，也是调用 plugin 的 identifier"""
    parameters: PropertyTree
    """指定参数列表，代码调用时所需，但不在由 Module 定义的参数列表中的参数。 """

    commit: str
    version: str
    copyright: str
    repository: str
    output_flag: array_type
    library: List[Library]

    def __str__(self) -> str:
        return f"{self.name} [{self.version or '0.0.0'}-{self.copyright or 'fytok'}]"

    def __repr__(self) -> str:
        desc = {
            "name": self.name,
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
    def _plugin_guess_name(cls, self: Module, cache, *args, **kwargs) -> str:
        pth = Path("code/name")
        plugin_name = (
            pth.get(cache, None)
            or pth.get(kwargs.get("default_value", _not_found_), None)
            or Path("code/metadata/default_value/name").get(cls, None)
        )

        return plugin_name

    def __init__(self, *args, **kwargs):
        if self.__class__ is Module or "_plugin_prefix" in vars(self.__class__):
            self.__class__.__dispatch_init__(None, self, *args, **kwargs)
            return
        
        super().__init__(*args, **kwargs)

        code = self._metadata.get("code", _not_found_)
        if code is not _not_found_:
            self.code.update(code)

        logger.info(f"Initialize module {self._plugin_prefix}{self.code or self.__class__.__name__} ")

    code: Code
    """ 对于 Module 的一般性说明。 
        @note code 在 __init__ 时由初始化参数定义，同时会根据 code.name 查找相应的 plugin 。"""

    @property
    def tag(self) -> str:
        return f"{FY_JOBID}/{self._plugin_prefix}{self.code}"

    def execute(self, *args, **kwargs) -> typing.Type[Actor]:
        logger.info(f"Execute module {self._plugin_prefix}{self.code}")
        return super().execute(*args, **kwargs)


class IDS(Module):
    """Base class of IDS"""

    ids_properties: IDSProperties
    """Interface Data Structure properties. This element identifies the node above as an IDS"""

    def __geometry__(self):
        return {}, {}

    def _repr_svg_(self) -> str:
        try:
            res = sp_view.display(self.__geometry__(), output="svg")
        except Exception as error:
            raise RuntimeError(f"{self}") from error
            # res = None
        return res


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


@sp_tree
class CoreRadialGrid:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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

    def remesh(self, rho_tor_norm=_not_found_, psi_norm=_not_found_) -> CoreRadialGrid:
        if (rho_tor_norm is None or rho_tor_norm is _not_found_) and (psi_norm is _not_found_ or psi_norm is None):
            rho_tor_norm_length = self.rho_tor_norm.size
            rho_tor_norm_axis = self.rho_tor_norm[0]
            rho_tor_norm_bdry = self.rho_tor_norm[-1]
            rho_tor_norm = np.linspace(rho_tor_norm_axis, rho_tor_norm_bdry, rho_tor_norm_length)
            psi_norm = self.psi_norm
        elif is_array(rho_tor_norm) and (psi_norm is _not_found_ or psi_norm is None):
            psi_norm = Function(self.rho_tor_norm, self.psi_norm)(rho_tor_norm)

        elif is_array(psi_norm) and (rho_tor_norm is None or rho_tor_norm is _not_found_):
            rho_tor_norm = Function(self.psi_norm, self.rho_tor_norm)(psi_norm)

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
    rho_tor_boundary: float
    rho_tor_norm: array_type

    @sp_property
    def psi(self) -> array_type:
        return self.psi_norm * (self.psi_boundary - self.psi_axis) + self.psi_axis

    @sp_property
    def rho_tor(self) -> array_type:
        return self.rho_tor_norm * self.rho_tor_boundary

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
