from __future__ import annotations
import collections
import functools
import typing
from dataclasses import dataclass
from enum import IntFlag
import numpy as np
from spdm.data.Path import Path, update_tree, merge_tree
from spdm.data.Actor import Actor
from spdm.data.AoS import AoS
from spdm.data.Field import Field
from spdm.data.Expression import Expression, zero
from spdm.data.Function import Function
from spdm.data.HTree import Dict, HTree, List
from spdm.data.Signal import Signal, SignalND

from spdm.data.sp_property import SpTree, sp_property, sp_tree, PropertyTree
from spdm.data.TimeSeries import TimeSeriesAoS, TimeSlice
from spdm.geometry.Curve import Curve
from spdm.utils.typing import array_type, is_array, as_array
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

    module_path: str
    """模块路径， 可用于 import 模块"""

    commit: str
    version: str
    copyright: str = ""
    repository: str
    output_flag: array_type
    library: List[Library]
    parameters: PropertyTree
    """指定参数列表，代码调用时所需，但不在由 Module 定义的参数列表中的参数。 """

    def __str__(self) -> str:
        return "-".join(
            [s for s in [self.module_path, self.version, self.copyright] if s not in (_not_found_, None, "")]
        )

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
    _plugin_registry = {}

    @classmethod
    def _plugin_guess_name(cls, self: Module, cache, *args, **kwargs) -> str:
        pth = Path("code/name")
        plugin_name = pth.get(cache, None) or pth.get(kwargs, None)
        if plugin_name is None:
            plugin_name = Path("code/metadata/default_value/name").get(cls, None)
        return plugin_name

    def __init__(self, *args, **kwargs):
        if self.__class__ is Module or "_plugin_prefix" in vars(self.__class__):
            prev_cls = self.__class__
            self.__class__.__dispatch_init__(None, self, *args, **kwargs)
            if self.__class__ is not prev_cls:
                return

        super().__init__(*args, **kwargs)

        if not self.code.name:
            self.code.name = self.__class__.__name__

        self.code.module_path = f"{self.__class__._plugin_prefix}{self.code.name}"

        logger.info(f"Initialize module {self.code} ")

    code: Code
    """ 对于 Module 的一般性说明。 
        @note code 在 __init__ 时由初始化参数定义，同时会根据 code.name 查找相应的 plugin 。"""

    @property
    def tag(self) -> str:
        return f"{FY_JOBID}/{self.code.module_path}"

    def execute(self, current: TimeSlice, *previous: TimeSlice) -> typing.Type[TimeSlice]:
        logger.info(f"Execute module {self.code.module_path}")
        return super().execute(current, *previous)


class IDS(Module):
    """Base class of IDS"""

    _plugin_prefix = f"fytok.modules."

    ids_properties: IDSProperties

    """Interface Data Structure properties. This element identifies the node above as an IDS"""

    def __geometry__(self):
        return {}

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

        # if self.fetch_cache("psi_norm", _not_found_) is _not_found_:
        #     psi = self.fetch_cache("psi", _not_found_)
        #     if psi is _not_found_:
        #         raise RuntimeError(f"Missing 'psi_norm' and 'psi' ! {args} {kwargs} ")
        #     psi = as_array(psi)
        #     self._cache["psi_axis"] = psi_axis = psi[0]
        #     self._cache["psi_boundary"] = psi_boundary = psi[-1]
        #     self._cache["psi_norm"] = (psi - psi_axis) / (psi_boundary - psi_axis)
        #     self._cache["psi"] = psi

        # if self.fetch_cache("rho_tor_norm", _not_found_) is _not_found_:
        #     rho_tor = self.fetch_cache("rho_tor", _not_found_)
        #     if rho_tor is _not_found_:
        #         raise RuntimeError(f"Missing 'rho_tor_norm' and 'rho_tor' ! ")
        #     rho_tor = as_array(rho_tor)
        #     self._cache["rho_tor_boundary"] = rho_tor[-1]
        #     self._cache["rho_tor_norm"] = rho_tor / rho_tor[-1]

    def __copy__(self) -> CoreRadialGrid:
        return CoreRadialGrid(
            {
                "psi_norm": self.psi_norm,
                "rho_tor_norm": self.rho_tor_norm,
                "psi_axis": self.psi_axis,
                "psi_boundary": self.psi_boundary,
                "rho_tor_boundary": self.rho_tor_boundary,
            }
        )

    def __serialize__(self, dumper=None):
        return HTree._do_serialize(
            {
                "psi_norm": self.psi_norm,
                "rho_tor_norm": self.rho_tor_norm,
                "psi_axis": self.psi_axis,
                "psi_boundary": self.psi_boundary,
                "rho_tor_boundary": self.rho_tor_boundary,
            },
            dumper,
        )

    def remesh(self, *args, **kwargs) -> CoreRadialGrid:
        """Duplicate the grid with new rho_tor_norm or psi_norm"""
        if len(args) == 0 or args[0] is _not_found_ or args[0] is None:
            grid = kwargs
        elif isinstance(args[0], dict):
            grid = collections.ChainMap(args[0], kwargs)
        elif isinstance(args[0], array_type):
            grid = collections.ChainMap({"rho_tor_norm": args[0]}, kwargs)
        else:
            raise TypeError(f"Invalid type of argument {args} {kwargs}")

        rho_tor_norm = grid.get("rho_tor_norm", _not_found_)
        psi_norm = grid.get("psi_norm", _not_found_)

        if rho_tor_norm is None or rho_tor_norm is _not_found_:
            if psi_norm is _not_found_ or psi_norm is None:
                psi_norm = self.psi_norm
                rho_tor_norm = self.rho_tor_norm
            else:
                rho_tor_norm = Function(self.psi_norm, self.rho_tor_norm)(psi_norm)
        elif psi_norm is _not_found_ or psi_norm is None:
            psi_norm = Function(self.rho_tor_norm, self.psi_norm)(rho_tor_norm)

        return CoreRadialGrid(
            {
                "rho_tor_norm": rho_tor_norm,
                "psi_norm": psi_norm,
                "psi_axis": self.psi_axis,
                "psi_boundary": self.psi_boundary,
                "rho_tor_boundary": self.rho_tor_boundary,
            }
        )

    psi_axis: float
    psi_boundary: float
    rho_tor_boundary: float

    psi_norm: array_type
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


@sp_tree
class CoreVectorComponents(SpTree):
    """Vector components in predefined directions"""

    radial: Expression = zero
    """ Radial component"""

    diamagnetic: Expression = zero
    """ Diamagnetic component"""

    parallel: Expression = zero
    """ Parallel component"""

    poloidal: Expression = zero
    """ Poloidal component"""

    toroidal: Expression = zero
    """ Toroidal component"""


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
