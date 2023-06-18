import functools
import typing
from dataclasses import dataclass

import numpy as np
from fytok._imas.lastest.utilities import (_T_core_radial_grid,
                                           _T_rz0d_dynamic_aos,
                                           _T_rz1d_dynamic_aos)
from spdm.data.Dict import Dict
from spdm.data.Entry import Entry
from spdm.data.Expression import Expression
from spdm.data.Function import Function, function_like
from spdm.data.Node import Node
from spdm.data.sp_property import SpDict, sp_property
from spdm.utils.logger import logger
from spdm.utils.tags import _not_found_
from spdm.utils.typing import array_type
from spdm.geometry.CubicSplineCurve import CubicSplineCurve

_T = typing.TypeVar("_T")

RZTuple1D = _T_rz1d_dynamic_aos
RZTuple = _T_rz0d_dynamic_aos
# class RZTuple(_T_rz1d_dynamic_aos):
#     r = sp_property(type="dynamic", units="m", ndims=1, data_type=float)
#     z = sp_property(type="dynamic", units="m", ndims=1, data_type=float)
# CoreRadialGrid = _T_core_radial_grid


class CurveRZ(SpDict, CubicSplineCurve):

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1 and not isinstance(args[0], array_type):
            d = args[0]
            args = ()
        else:
            d = None

        super().__init__(d, **kwargs)

        if len(args) == 0:
            args = [np.vstack([super().get("r"), super().get("z")])]

        CubicSplineCurve.__init__(self, *args)

    @sp_property
    def r(self) -> array_type: return self.points[0]

    @sp_property
    def z(self) -> array_type: return self.points[1]


@dataclass
class RZTuple_:
    r: np.ndarray | Expression
    z: np.ndarray | Expression


class CoreRadialGrid(_T_core_radial_grid):
    """1D radial grid for core profiles"""

    @functools.cached_property
    def r0(self) -> float: return self.get("../../vacuum_toroidal_field/r0")

    @functools.cached_property
    def b0(self) -> float:
        time = self.get("../time", 0.0)
        return self.get("../../vacuum_toroidal_field/b0")(time)

    rho_tor_norm: np.ndarray = sp_property(type="dynamic", coordinate1="1...N", units="-")

    rho_tor: Function[float] = sp_property(type="dynamic", coordinate1="../rho_tor_norm", units="m")
    """ Toroidal flux coordinate"""

    @sp_property
    def rho_pol_norm(self) -> Function[float]: return np.sqrt(self.psi_norm)

    psi_norm: np.ndarray = sp_property(coordinate1="../rho_tor_norm", units="-")

    @sp_property()
    def psi(self) -> Function[float]:
        return self.psi_norm * (self.psi_boundary - self.psi_magnetic_axis)+- self.psi_magnetic_axis

    volume: Function[float] = sp_property()

    area: Function[float] = sp_property()

    surface: Function[float] = sp_property()

    psi_magnetic_axis: float = sp_property()

    psi_boundary: float = sp_property()
