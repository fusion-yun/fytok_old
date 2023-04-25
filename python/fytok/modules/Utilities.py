import typing
from dataclasses import dataclass

import numpy as np
from _imas.utilities import _T_rz1d_dynamic_aos
from spdm.data.Dict import Dict
from spdm.data.Entry import Entry
from spdm.data.Field import Field
from spdm.data.Function import Function, function_like
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property

_T = typing.TypeVar("_T")


class RZTuple(_T_rz1d_dynamic_aos):
    r = sp_property(type="dynamic", units="m", ndims=1, data_type=float)
    z = sp_property(type="dynamic", units="m", ndims=1, data_type=float)


class RadialGrid(Dict[Node]):
    """ Radial grid """

    def remesh(self, label: str = "psi_norm", new_axis: np.ndarray = None, ):

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

        return RadialGrid(
            r0=self.r0,
            b0=self.b0,
            psi_axis=self.psi_axis,
            psi_boundary=self.psi_boundary,
            rho_tor_boundary=self.rho_tor_boundary,
            psi_norm=function_like(axis,  self.psi_norm)(new_axis) if label != "psi_norm" else new_axis,
            rho_tor_norm=function_like(axis,  self.rho_tor_norm)(new_axis) if label != "rho_tor_norm" else new_axis,
            # rho_pol_norm=Function(axis,  self.rho_pol_norm)(new_axis) if label != "rho_pol_norm" else new_axis,
            # area=Function(axis,  self.area)(new_axis) if label != "area" else new_axis,
            # surface=Function(axis,  self.surface)(new_axis) if label != "surface" else new_axis,
            # volume=Function(axis,  self.volume)(new_axis) if label != "volume" else new_axis,
            dvolume_drho_tor=function_like(axis,  self.dvolume_drho_tor)(new_axis),
        )

    r0: float = sp_property()

    b0: float = sp_property()

    psi_axis: float = sp_property()

    psi_magnetic_axis: float = sp_property()

    psi_boundary: float = sp_property()

    rho_tor_boundary: float = sp_property()

    psi_norm:  np.ndarray = sp_property()

    psi: np.ndarray = sp_property(lambda self: self.psi_norm * (self.psi_boundary-self.psi_axis)+self.psi_axis)

    rho_tor_norm: np.ndarray = sp_property()

    rho_tor: np.ndarray = sp_property(lambda self: self.rho_tor_norm*self.rho_tor_boundary)

    rho_pol_norm:  np.ndarray = sp_property()

    area:  np.ndarray = sp_property()

    surface:  np.ndarray = sp_property()

    volume:  np.ndarray = sp_property()

    dvolume_drho_tor: np.ndarray = sp_property()
