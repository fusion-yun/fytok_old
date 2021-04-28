import collections
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import arctan2, cos, sin, sqrt
from packaging import version
from scipy.optimize import fsolve, root_scalar
from spdm.data.Coordinates import Coordinates
from spdm.data.Field import Field
from spdm.data.Function import Function
from spdm.data.mesh import Mesh
from spdm.data.mesh.CurvilinearMesh import CurvilinearMesh
from spdm.data.mesh.RectilinearMesh import RectilinearMesh

from spdm.data.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.utilities import try_get


class CoordinateSystem(AttributeTree, Coordinates):
    r"""
        Definition of the 2D grid

        grid.dim1           :
                First dimension values  [mixed]
        grid.dim2           :
                Second dimension values  [mixed]
        grid.volume_element :
                Elementary plasma volume of plasma enclosed in the cell formed by the nodes [dim1(i) dim2(j)], [dim1(i+1) dim2(j)], [dim1(i) dim2(j+1)] and [dim1(i+1) dim2(j+1)]  [m^3]

        grid_type.name      :
                name

        grid_type.index     :
                0xFF (rho theta)
                theta
                0x0?  : rho= psi,  poloidal flux function
                0x1?  : rho= V, volume within surface
                0x2?  : rho= sqrt(V)
                0x3?  : rho= Phi, toroidal flux with in surface
                0x4?  : rho= sqrt(Phi/B0/pi)
                theta
                0x?0  : poloidal angle
                0x?1  : constant arch length
                0x?2  : straight filed lines
                0x?3  : constant area
                0x?4  : constant volume
    """

    def __init__(self,  *args, mesh=None, jacobian=None, **kwargs):
        AttributeTree.__init__(self, *args)

        if mesh is not None:
            Coordinates.__init__(self, mesh=mesh, **kwargs)
        else:
            grid_name = self.grid_type.name or "rectangle"
            grid_index = self.grid_type.index or 1
            if grid_index == 1:
                dim1 = self["grid.dim1"]
                dim2 = self["grid.dim2"]
                mesh = RectilinearMesh(dim1, dim2)
            elif grid_index > 10:
                u = self["grid.dim1"]
                v = self["grid.dim2"]
                r = self["r"]
                z = self["z"]
                mesh = CurvilinearMesh([r, z], [u, v], cycle=[False, True])

            Coordinates.__init__(self, mesh=mesh, name=grid_name, **kwargs)
        self._jacobian = jacobian

    # @cached_property
    # def _metric(self):
    #     jacobian = np.full(self._shape, np.nan)
    #     tensor_covariant = np.full([*self._shape, 3, 3], np.nan)
    #     tensor_contravariant = np.full([*self._shape, 3, 3], np.nan)

    #     # TODO: not complete

    #     return AttributeTree({
    #         "jacobian": jacobian,
    #         "tensor_covariant": tensor_covariant,
    #         "tensor_contravariant": tensor_contravariant
    #     })

    @cached_property
    def dl(self):
        return np.asarray([self.mesh.axis(idx, axis=0).geo_object.dl(self.mesh.uv[1]) for idx in range(self.mesh.shape[0])])

    @cached_property
    def r(self):
        """	Values of the major radius on the grid  [m]"""
        return self.mesh.xy[0]

    @cached_property
    def z(self):
        """Values of the Height on the grid  [m]"""
        return self.mesh.xy[1]

    @cached_property
    def jacobian(self):
        """	Absolute value of the jacobian of the coordinate system  [mixed]"""
        return self._metric.jacobian

    @cached_property
    def vprime(self):
        r"""
            .. math:: V^{\prime} =  2 \pi  \int{ R / |\nabla \psi| * dl }
            .. math:: V^{\prime}(psi)= 2 \pi  \int{ dl * R / |\nabla \psi|}
        """
        return self.surface_integral() * (2*scipy.constants.pi)

    def surface_integral(self, J=None, *args, without_jacobian=False, **kwargs):
        r"""
            .. math:: \left\langle \alpha\right\rangle \equiv\frac{2\pi}{V^{\prime}}\oint\alpha\frac{Rdl}{\left|\nabla\psi\right|}
        """
        if J is None:
            J = self._jacobian
        elif not without_jacobian:
            J = J * self._jacobian
        return np.sum(0.5*(np.roll(J, 1, axis=1)+J) * self.dl, axis=1)

    def surface_average(self,   *args, **kwargs):
        return self.surface_integral(*args, **kwargs) / self.vprime * (2*scipy.constants.pi)

    @cached_property
    def tensor_covariant(self):
        """Covariant metric tensor on every point of the grid described by grid_type  [mixed]. """
        return self._metric.tensor_covariant

    @cached_property
    def tensor_contravariant(self):
        """Contravariant metric tensor on every point of the grid described by grid_type  [mixed]"""
        return self._metric.tensor_contravariant
