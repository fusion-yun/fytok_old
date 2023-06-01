import typing
from dataclasses import dataclass

import numpy as np
from fytok._imas.lastest.utilities import (_T_core_radial_grid,
                                           _T_rz0d_dynamic_aos,
                                           _T_rz1d_dynamic_aos)
from spdm.data.Dict import Dict
from spdm.data.Entry import Entry
from spdm.data.Function import Function, function_like
from spdm.data.Expression import Expression

from spdm.data.Node import Node
from spdm.data.Profile import Profile
from spdm.data.sp_property import SpDict, sp_property
from spdm.utils.logger import logger
from spdm.utils.tags import _not_found_

_T = typing.TypeVar("_T")

RZTuple1D = _T_rz1d_dynamic_aos
RZTuple = _T_rz0d_dynamic_aos
# class RZTuple(_T_rz1d_dynamic_aos):
#     r = sp_property(type="dynamic", units="m", ndims=1, data_type=float)
#     z = sp_property(type="dynamic", units="m", ndims=1, data_type=float)
# CoreRadialGrid = _T_core_radial_grid


@dataclass
class RZTuple_:
    r: np.ndarray | Expression[float]
    z: np.ndarray | Expression[float]


class CoreRadialGrid(_T_core_radial_grid):
    """1D radial grid for core profiles"""

    def remesh(self, label: str = "psi_norm", new_axis: np.ndarray = None, ):

        logger.warning("TODO: incorrect implement! need fix!")

        axis = self._as_child(label)

        if isinstance(axis, np.ndarray) and isinstance(new_axis, np.ndarray) \
                and axis.shape == new_axis.shape and np.allclose(axis, new_axis):
            return self

        if new_axis is None:
            new_axis = np.linspace(axis[0], axis[-1], len(axis))
        elif isinstance(new_axis, int):
            new_axis = np.linspace(0, 1.0, new_axis)
        elif not isinstance(new_axis, np.ndarray):
            raise TypeError(new_axis)

        return CoreRadialGrid(
            {
                "psi_magnetic_axis": self.psi_magnetic_axis,
                "psi_boundary":     self.psi_boundary,
                "psi":              function_like(self.psi, axis)(new_axis) if label != "psi_norm" else new_axis,
                "rho_tor_norm":     new_axis,
                # rho_pol_norm=Function(  self.rho_pol_norm,axis)(new_axis) if label != "rho_pol_norm" else new_axis,
                # area=     Function(self.area,axis)(new_axis) if label != "area" else new_axis,
                # surface=  Function(self.surface,axis)(new_axis) if label != "surface" else new_axis,
                # volume=   Function(self.volume,axis)(new_axis) if label != "volume" else new_axis,
                "dvolume_drho_tor": function_like(self.dvolume_drho_tor, axis)(new_axis),
            },
            parent=self._parent
        )

    rho_tor_norm: np.ndarray = sp_property(type="dynamic", coordinate1="1...N", units="-")

    rho_tor: Profile[float] = sp_property(type="dynamic", coordinate1="../rho_tor_norm", units="m")
    """ Toroidal flux coordinate"""

    @sp_property
    def rho_pol_norm(self) -> Profile[float]: return np.sqrt(self.psi_norm)

    @sp_property(coordinate1="../rho_tor_norm")
    def psi_norm(self) -> Profile[float]:
        v = super().get("psi_norm", _not_found_, raw=True)
        if v is not _not_found_:
            return v
        else:
            return (self.psi - self.psi_magnetic_axis) / (self.psi_boundary - self.psi_magnetic_axis)

    psi: Profile[float] = sp_property()

    volume: Profile[float] = sp_property()

    area: Profile[float] = sp_property()

    surface: Profile[float] = sp_property()

    psi_magnetic_axis: float = sp_property()

    psi_boundary: float = sp_property()
