from functools import cached_property

import numpy as np
from _imas.equilibrium import (_T_equilibrium, _T_equilibrium_boundary,
                               _T_equilibrium_global_quantities,
                               _T_equilibrium_profiles_1d,
                               _T_equilibrium_profiles_2d)
from scipy import constants
from spdm.common.tags import _not_found_, _undefined_
from spdm.data.Dict import Dict
from spdm.data.Field import Field
from spdm.data.Function import Function, function_like
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property
from spdm.util.logger import logger
from spdm.util.misc import convert_to_named_tuple, try_get

from .MagneticCoordSystem import MagneticCoordSystem, RadialGrid
from .PFActive import PFActive
from .Wall import Wall
from .Utilities import RZTuple

TOLERANCE = 1.0e-6
EPS = np.finfo(float).eps

TWOPI = 2.0*constants.pi


class EquilibriumGlobalQuantities(_T_equilibrium_global_quantities):

    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    @cached_property
    def magnetic_axis(self):
        """Magnetic axis position and toroidal field	structure"""
        return convert_to_named_tuple(self._coord.magnetic_axis)

    @sp_property
    def x_points(self):
        _, x = self._coord.critical_points
        return x

    @sp_property
    def psi_axis(self) -> float:
        """Poloidal flux at the magnetic axis[Wb]."""
        o, _ = self._coord.critical_points
        return o[0].psi

    @sp_property
    def psi_boundary(self) -> float:
        """Poloidal flux at the selected plasma boundary[Wb]."""
        _, x = self._coord.critical_points
        if len(x) > 0:
            return x[0].psi
        else:
            raise ValueError(f"No x-point")


class EquilibriumProfiles1D(_T_equilibrium_profiles_1d):

    @cached_property
    def _predefined_psi_norm(self):
        psi = self._entry.get("psi", None)
        return (psi-psi[0])/(psi[-1]-psi[0])

    @sp_property
    def pressure(self) -> Function:
        return function_like(self._predefined_psi_norm, self._entry.get("pressure", None))

    @sp_property
    def dpressure_dpsi(self) -> Function:
        return function_like(self._predefined_psi_norm,  self._entry.get("dpressure_dpsi", None))

    @property
    def pprime(self) -> Function:
        return self.dpressure_dpsi

    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    @sp_property
    def ffprime(self) -> Function:
        """	Derivative of F w.r.t. Psi, multiplied with F[T ^ 2.m ^ 2/Wb]. """
        return function_like(self._coord.psi_norm, self._coord.ffprime)

    @sp_property
    def f_df_dpsi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.ffprime)

    @sp_property
    def fpol(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.fpol)

    @sp_property
    def f(self) -> Function:
        return self.fpol

    @sp_property
    def plasma_current(self) -> Function:
        return self.gm2 * self.dvolume_drho_tor / self.dpsi_drho_tor/constants.mu_0

    @sp_property
    def j_tor(self) -> Function:
        return self.plasma_current.derivative() / (self._coord.psi_boundary - self._coord.psi_axis)/self.dvolume_dpsi * self._coord.r0

    @sp_property
    def j_parallel(self) -> Function:
        fvac = self._coord._fvac
        d = np.asarray(Function(np.asarray(self.volume), np.asarray(fvac*self.plasma_current/self.fpol)).derivative())
        return self._coord.r0*(self.fpol / fvac)**2 * d

    @sp_property
    def psi_norm(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.psi_norm)

    @sp_property
    def psi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.psi)

    @sp_property
    def dphi_dpsi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.dphi_dpsi)

    @sp_property
    def q(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.q)

    @sp_property
    def phi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.phi)

    @sp_property
    def rho_tor(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.rho_tor)

    @sp_property
    def rho_tor_norm(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.rho_tor_norm)

    @sp_property
    def drho_tor_dpsi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.drho_tor_dpsi)

    @sp_property
    def rho_volume_norm(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.rho_volume_norm)

    @sp_property
    def area(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.area)

    @sp_property
    def darea_dpsi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.darea_dpsi)

    @sp_property
    def darea_drho_tor(self) -> Function	:
        return function_like(self._coord.psi_norm, self._coord.darea_drho_tor)

    @sp_property
    def surface(self):
        return function_like(self._coord.psi_norm, self._coord.surface)

    @sp_property
    def volume(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.volume)

    @sp_property
    def dvolume_dpsi(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.dvolume_dpsi)

    @sp_property
    def dpsi_drho_tor(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.dpsi_drho_tor)

    @sp_property
    def dpsi_drho_tor_norm(self) -> Function:
        return self.dpsi_drho_tor*self.rho_tor[-1]

    @sp_property
    def dvolume_drho_tor(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.dvolume_drho_tor)

    @cached_property
    def shape_property(self) -> MagneticCoordSystem.ShapeProperty:
        return self._coord.shape_property()

    @sp_property
    def geometric_axis(self) -> RZTuple[Function]:
        gaxis = self.shape_property.geometric_axis
        return RZTuple[Function](function_like(self._coord.psi_norm, gaxis.r),
                                 function_like(self._coord.psi_norm, gaxis.z))

    @sp_property
    def minor_radius(self) -> Function:
        return function_like(self._coord.psi_norm, self.shape_property.minor_radius)

    @sp_property
    def r_inboard(self) -> Function:
        return function_like(self._coord.psi_norm, self.shape_property.r_inboard)

    @sp_property
    def r_outboard(self) -> Function:
        return function_like(self._coord.psi_norm, self.shape_property.r_outboard)

    # @sp_property
    # def elongation(self) -> Function:
    #     """Elongation. {dynamic}[-]"""
    #     return self.shape_property.elongation
    @sp_property
    def elongation(self) -> Function:
        return function_like(self._coord.psi_norm, self.shape_property.elongation)

    @sp_property
    def triangularity(self) -> Function	:
        return function_like(self._coord.psi_norm, self.shape_property.triangularity)

    @sp_property
    def triangularity_upper(self) -> Function	:
        return function_like(self._coord.psi_norm, self.shape_property.triangularity_upper)

    @sp_property
    def triangularity_lower(self) -> Function:
        return function_like(self._coord.psi_norm, self.shape_property.triangularity_lower)

    @sp_property
    def gm1(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm1)

    @sp_property
    def gm2(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm2)

    @sp_property
    def gm3(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm3)

    @sp_property
    def gm4(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm4)

    @sp_property
    def gm5(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm5)

    @sp_property
    def gm6(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm6)

    @sp_property
    def gm7(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm7)

    @sp_property
    def gm8(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm8)

    @sp_property
    def gm9(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.gm9)

    @sp_property
    def magnetic_shear(self) -> Function:
        return function_like(self._coord.psi_norm, self._coord.magnetic_shear)

    @sp_property
    def trapped_fraction(self, value) -> Function:
        """Trapped particle fraction[-]
            Tokamak 3ed, 14.10
        """
        if value is _not_found_:
            epsilon = self.rho_tor/self._coord.r0
            value = np.asarray(1.0 - (1-epsilon)**2/np.sqrt(1.0-epsilon**2)/(1+1.46*np.sqrt(epsilon)))
        return function_like(self._coord.psi_norm, value)


class EquilibriumProfiles2D(_T_equilibrium_profiles_2d):
    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    @cached_property
    def grid(self):
        return convert_to_named_tuple(self.get("grid", {}))

    @cached_property
    def grid_type(self):
        return convert_to_named_tuple(self.get("grid_type", {}))

    @cached_property
    def psi(self) -> Field:
        return self._coord._psirz  # (self._coord.r,self._coord.z)

    @sp_property
    def r(self) -> np.ndarray:
        """Values of the major radius on the grid  [m] """
        return self._coord.r

    @sp_property
    def z(self) -> np.ndarray:
        """Values of the Height on the grid  [m] """
        return self._coord.z

    # @sp_property
    # def psi(self):
    #     """Values of the poloidal flux at the grid in the poloidal plane  [Wb]. """
    #     return self.apply_psifunc(lambda p: p, unit="Wb")

    @sp_property
    def phi(self):
        return self.apply_psifunc("phi")

    @sp_property
    def j_tor(self):
        return self.apply_psifunc("j_tor")

    @sp_property
    def j_parallel(self):
        return self.apply_psifunc("j_parallel")

    @sp_property
    def b_field_r(self):
        return Field(self._coord.Br, self._coord.r, self._coord.z, mesh_type="curvilinear")

    @sp_property
    def b_field_z(self):
        return Field(self._coord.Bz, self._coord.r, self._coord.z, mesh_type="curvilinear")

    @sp_property
    def b_field_tor(self):
        return Field(self._coord.Btor, self._coord.r, self._coord.z, mesh_type="curvilinear")


class EquilibriumBoundary(_T_equilibrium_boundary):

    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    @sp_property
    def type(self):
        """0 (limiter) or 1 (diverted)  """
        return 1

    @sp_property
    def outline(self) -> RZTuple:
        """RZ outline of the plasma boundary  """
        _, surf = next(self._coord.find_surface(self.psi, o_point=True))
        return RZTuple(surf.xyz[0], surf.xyz[1])

    @sp_property
    def x_point(self):
        _, xpt = self._parent.critical_points
        return xpt

    @sp_property
    def psi_axis(self) -> float:
        return self._coord.psi_axis

    @sp_property
    def psi_boundary(self) -> float:
        return self._coord.psi_boundary

    @sp_property
    def psi(self) -> float:
        """Value of the poloidal flux at which the boundary is taken  [Wb]"""
        return self.psi_norm*(self._coord.psi_boundary-self._coord.psi_axis)+self._coord.psi_axis

    psi_norm: float = sp_property(default_value=0.999)
    """Value of the normalized poloidal flux at which the boundary is taken (typically 99.x %),
            the flux being normalized to its value at the separatrix """

    @property
    def shape_property(self) -> MagneticCoordSystem.ShapeProperty:
        return self._coord.shape_property(self.psi_norm)

    @sp_property
    def geometric_axis(self) -> RZTuple:
        """RZ position of the geometric axis (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the boundary)"""
        return self.shape_property.geometric_axis

    @sp_property
    def minor_radius(self) -> float:
        """Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary) [m]	"""
        return self.shape_property.minor_radius

    @sp_property
    def elongation(self) -> float:
        """Elongation of the plasma boundary. [-]	"""
        return self.shape_property.elongation

    @sp_property
    def elongation_upper(self) -> float:
        """Elongation(upper half w.r.t. geometric axis) of the plasma boundary. [-]	"""
        return self.shape_property.elongation_upper

    @sp_property
    def elongation_lower(self) -> float:
        """Elongation(lower half w.r.t. geometric axis) of the plasma boundary. [-]	"""
        return self.shape_property.elongation_lower

    @sp_property
    def triangularity(self) -> float:
        """Triangularity of the plasma boundary. [-]	"""
        return self.shape_property.triangularity

    @sp_property
    def triangularity_upper(self) -> float:
        """Upper triangularity of the plasma boundary. [-]	"""
        return self.shape_property.triangularity_upper

    @sp_property
    def triangularity_lower(self) -> float:
        """Lower triangularity of the plasma boundary. [-]"""
        return self.shape_property.triangularity_lower

    @sp_property
    def strike_point(self)	:
        """Array of strike points, for each of them the RZ position is given	struct_array [max_size=unbounded]	1- 1...N"""
        return NotImplemented

    @sp_property
    def active_limiter_point(self):
        """	RZ position of the active limiter point (point of the plasma boundary in contact with the limiter)"""
        return NotImplemented


class EquilibriumBoundarySeparatrix(Dict[Node]):

    @property
    def _coord(self) -> MagneticCoordSystem:
        return self._parent.coordinate_system

    @sp_property
    def type(self):
        """0 (limiter) or 1 (diverted)  """
        return 1

    @sp_property
    def outline(self) -> RZTuple:
        """RZ outline of the plasma boundary  """
        _, surf = next(self._coord.find_surface_by_psi_norm(1.0, o_point=None))
        return RZTuple(*surf.xyz)

    @sp_property
    def psi_axis(self) -> float:
        return self._coord.psi_axis

    @sp_property
    def psi_boundary(self) -> float:
        return self._coord.psi_boundary

    @sp_property
    def psi(self) -> float:
        return self._coord.psi_norm*(self._coord.psi_boundary-self._coord.psi_axis)+self._coord.psi_axis

    psi_norm: float = sp_property(default_value=1.0)


class Equilibrium(_T_equilibrium):
    r"""
        Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.

        Reference:
            - O. Sauter and S. Yu Medvedev, "Tokamak coordinate conventions: COCOS", Computer Physics Communications 184, 2 (2013), pp. 293--302.

        COCOS  11
        ```{text}
            Top view
                     ***************
                    *               *
                   *   ***********   *
                  *   *           *   *
                 *   *             *   *
                 *   *             *   *
             Ip  v   *             *   ^  \phi
                 *   *    Z o--->R *   *
                 *   *             *   *
                 *   *             *   *
                 *   *     Bpol    *   *
                  *   *     o     *   *
                   *   ***********   *
                    *               *
                     ***************
                       Bpol x
                    Poloidal view
                ^Z
                |
                |       ************
                |      *            *
                |     *         ^    *
                |     *   \rho /     *
                |     *       /      *
                +-----*------X-------*---->R
                |     *  Ip, \phi   *
                |     *              *
                |      *            *
                |       *****<******
                |       Bpol,\theta
                |
                    Cylindrical coordinate      : $(R,\phi,Z)$
            Poloidal plane coordinate   : $(\rho,\theta,\phi)$
        ```
    """

    def refresh(self,  *args,
                wall: Wall = _undefined_,
                pf_active: PFActive = _undefined_,
                core_profiles=_undefined_,
                **kwargs):
        super().refresh(*args, **kwargs)

        # self.profiles_1d.pressure = core_profiles.profiles_1d.pressure
        # self.profiles_1d.pressure = core_profiles.profiles_1d.fpol

        # # call Eq solver
        # psi_2d = self._eq_solver(
        #     {
        #         "vacuum_toroidal_field": self.vacuum_toroidal_field,
        #         "global_quantities": {"ip": self.global_quantities.ip,
        #                               "bet": self.global_quantities.betn
        #                               }
        #     }
        # )
        # self.profiles_2d.psi = psi_2d
        # return {
        #     "psi": psi_2d,
        #     "fpol": fpol,
        #     "pprime": pprime,
        # }
        return

    profiles_1d: EquilibriumProfiles1D = sp_property()

    profiles_2d: EquilibriumProfiles2D = sp_property()

    global_quantities: EquilibriumGlobalQuantities = sp_property()

    boundary: EquilibriumBoundary = sp_property()

    boundary_separatrix: EquilibriumBoundarySeparatrix = sp_property()

    @sp_property
    def coordinate_system(self, desc) -> MagneticCoordSystem:
        psirz = self.profiles_2d.get("psi", None)

        if not isinstance(psirz, Field):
            psirz = Field(psirz,
                          self.profiles_2d.grid.dim1,
                          self.profiles_2d.grid.dim2,
                          mesh="rectilinear")

        psi_1d = self.profiles_1d._entry.get("psi")
        fpol_1d = self.profiles_1d._entry.get("f", _not_found_)
        if not isinstance(psi_1d, np.ndarray) or len(psi_1d) != len(fpol_1d):
            psi_1d = np.linspace(0, 1.0, len(fpol_1d))

        if isinstance(psi_1d, np.ndarray):
            psi_1d = (psi_1d-psi_1d[0])/(psi_1d[-1]-psi_1d[0])

        # pprime_1d = self.profiles_1d._entry.get("dpressure_dpsi", None)

        res = MagneticCoordSystem(
            psirz=psirz,
            B0=self.vacuum_toroidal_field.b0,
            R0=self.vacuum_toroidal_field.r0,
            Ip=self.global_quantities.ip,
            fpol=function_like(psi_1d, fpol_1d),
            # pprime=self.profiles_1d._entry.get("dpressure_dpsi", None),
            # fpol=function_like(psi_norm, self.profiles_1d._entry.get("f", None)),
            # pprime=function_like(psi_norm, self.profiles_1d._entry.get("dpressure_dpsi", None)),
            **desc
        )
        return res

    @property
    def radial_grid(self) -> RadialGrid:
        return self.coordinate_system.radial_grid

    def plot(self, axis=None, /,
             scalar_field=[],
             vector_field=[],
             boundary=True,
             separatrix=True,
             contours=True,
             oxpoints=True,
             **kwargs):
        """
            plot o-point,x-point,lcfs,separatrix and contour of psi
        """
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as error:
            logger.error(f"Can not load matplotlib! [{error}]")
            return
        if axis is None:
            axis = plt.gca()

        # R = self.profiles_2d.r
        # Z = self.profiles_2d.z
        # psi = self.profiles_2d.psi(R, Z)

        # axis.contour(R[1:-1, 1:-1], Z[1:-1, 1:-1], psi[1:-1, 1:-1], levels=levels, linewidths=0.2)
        if oxpoints is not False:
            o_point, x_point = self.coordinate_system.critical_points

            axis.plot(o_point[0].r,
                      o_point[0].z,
                      'g+',
                      linewidth=0.5,
                      #   markersize=2,
                      label="Magnetic axis")

            if len(x_point) > 0:
                for idx, p in enumerate(x_point):
                    axis.plot(p.r, p.z, 'rx')
                    axis.text(p.r, p.z, idx,
                              horizontalalignment='center',
                              verticalalignment='center')

                axis.plot([], [], 'rx', label="X-Point")

        if boundary is not False:
            boundary_points = np.vstack([self.boundary.outline.r,
                                         self.boundary.outline.z]).T

            axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='solid',
                                       linewidth=0.5, fill=False, closed=True))

            axis.plot([], [], 'g-', label="Boundary")

        if separatrix is not False:
            # r0 = self._entry.get("boundary_separatrix.outline.r", None)
            # z0 = self._entry.get("boundary_separatrix.outline.z", None)
            # if r0 is not None and z0 is not None:
            #     axis.add_patch(plt.Polygon(np.vstack([r0, z0]).T, color='b', linestyle=':',
            #                                linewidth=1.0, fill=False, closed=True))

            boundary_points = np.vstack([self.boundary_separatrix.outline.r,
                                         self.boundary_separatrix.outline.z]).T

            axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='dashed',
                                       linewidth=0.5, fill=False, closed=False))
            axis.plot([], [], 'r--', label="Separatrix")

        if contours is not False:
            if contours is True:
                contours = 16
            self.coordinate_system.plot_contour(axis, contours)
            # if isinstance(contour, int):
            #     c_list = range(0, self.coordinate_system.mesh.shape[0], int(
            #         self.coordinate_system.mesh.shape[0]/contour+0.5))
            # elif isinstance(contour, collections.abc.Sequcence):
            #     c_list = contour
            # for idx in c_list:
            #     ax0 = self.coordinate_system.mesh.axis(idx, axis=0)

            #     if ax0.xy.shape[1] == 1:
            #         axis.add_patch(plt.Circle(ax0.xy[:, 0], radius=0.05, fill=False,color="b", linewidth=0.2))
            #     else:
            #         axis.add_patch(plt.Polygon(ax0.xy, fill=False, closed=True, color="b", linewidth=0.2))

        for s, opts in scalar_field:
            if s == "psirz":
                self.coordinate_system._psirz.plot(axis, **opts)
            else:
                if "." not in s:
                    sf = f"profiles_2d.{s}"
                # self.coordinate_system.norm_grad_psi
                sf = try_get(self, s, None)
                if isinstance(sf, Field):
                    sf.plot(axis, **opts)
                elif isinstance(sf, np.ndarray):
                    axis.contour(self.profiles_2d.r, self.profiles_2d.z, sf, **opts)
                else:
                    logger.error(f"Can not find field {sf} {type(sf)}!")

        for u, v, opts in vector_field:
            uf = self.profiles_2d[u]
            vf = self.profiles_2d[v]
            axis.streamplot(self.profiles_2d.grid.dim1,
                            self.profiles_2d.grid.dim2,
                            vf, uf, **opts)

        return axis
