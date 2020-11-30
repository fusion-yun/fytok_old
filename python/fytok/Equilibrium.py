import collections
import functools
import inspect
from functools import cached_property, lru_cache

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import arctan2, cos, sin, sqrt
import scipy.constants as constants
from scipy.interpolate import (RectBivariateSpline, SmoothBivariateSpline,
                               UnivariateSpline)
from scipy.optimize import root_scalar
from spdm.data.Entry import open_entry
from spdm.util.AttributeTree import AttributeTree, _next_
from spdm.util.Interpolate import (Interpolate1D, Interpolate2D, derivate,
                                   find_critical, find_root, integral,
                                   interpolate)
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module
from spdm.util.utilities import first_not_empty
from sympy import Point, Polygon
from spdm.util.Profiles import Profiles

from fytok.FluxSurface import FluxSurface
from fytok.PFActive import PFActive
from fytok.Wall import Wall

TOLERANCE = 1.0e-6


class Equilibrium(AttributeTree):
    r"""Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.

        Reference:
            - O. Sauter and S. Yu Medvedev, "Tokamak coordinate conventions: COCOS", Computer Physics Communications 184, 2 (2013), pp. 293--302.

        COCOS  11

    """
    #    Top view
    #             ***************
    #            *               *
    #           *   ***********   *
    #          *   *           *   *
    #         *   *             *   *
    #         *   *             *   *
    #     Ip  v   *             *   ^  \phi
    #         *   *    Z o--->R *   *
    #         *   *             *   *
    #         *   *             *   *
    #         *   *     Bpol    *   *
    #          *   *     o     *   *
    #           *   ***********   *
    #            *               *
    #             ***************
    #               Bpol x
    #            Poloidal view
    #        ^Z
    #        |
    #        |       ************
    #        |      *            *
    #        |     *         ^    *
    #        |     *   \rho /     *
    #        |     *       /      *
    #        +-----*------X-------*---->R
    #        |     *  Ip, \phi   *
    #        |     *              *
    #        |      *            *
    #        |       *****<******
    #        |       Bpol,\theta
    #        |
    #            Cylindrical coordinate      : (R,\phi,Z)
    #    Poloidal plane coordinate   : (\rho,\theta,\phi)

    IDS = "Equilibrium"

    @staticmethod
    def __new__(cls,   config,  *args, **kwargs):
        if cls is not Equilibrium:
            return super(Equilibrium, cls).__new__(cls)

        backend = str(config.engine or "FreeGS")
        n_cls = cls

        if backend != "":
            try:
                plugin_name = f"{__package__}.plugins.equilibrium.Plugin{backend}"
                n_cls = sp_find_module(plugin_name, fragment=f"Equilibrium{backend}")

            except ModuleNotFoundError as error:
                logger.debug(error)
                n_cls = cls

        return AttributeTree.__new__(n_cls)

    def __init__(self,   cache=None,  *args,  tokamak=None, psi_norm=129, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokamak = tokamak
        self._cache = cache
        if type(psi_norm) is int:
            self._psi_norm = np.linspace(0.0, 1.0, psi_norm)
        elif isinstance(psi_norm, np.ndarray):
            self._psi_norm = psi_norm
        else:
            raise TypeError(f"{psi_norm}")
        # ----------------------------------------------------------
        # self.__dict__["_grid_box"] = AttributeTree()

        # lim_r = self.tokamak.wall.limiter.outline.r
        # lim_z = self.tokamak.wall.limiter.outline.z

        # rmin = config.rmin or kwargs.get("rmin", None) or min(lim_r)
        # rmax = config.rmax or kwargs.get("rmax", None) or max(lim_r)
        # zmin = config.zmin or kwargs.get("zmin", None) or min(lim_z)
        # zmax = config.zmax or kwargs.get("zmax", None) or max(lim_z)
        # nr = config.nr or kwargs.get("nr", None) or 129
        # nz = config.nz or kwargs.get("nz", None) or 129
        # self._grid_box.dim1 = np.linspace(rmin-0.1*(rmax-rmin), rmax+0.1*(rmax-rmin), nr)
        # self._grid_box.dim2 = np.linspace(zmin-0.1*(zmax-zmin), zmax+0.1*(zmax-zmin), nz)

    @property
    def cache(self):
        if isinstance(self._cache, LazyProxy):
            self._cache = self._cache()
        elif isinstance(self._cache, AttributeTree):
            self._cache = self._cache
        else:
            self._cache = AttributeTree(self._cache)
        return self._cache

    @cached_property
    def vacuum_toroidal_field(self):
        return self.tokamak.vacuum_toroidal_field

    @property
    def time(self):
        return self.tokamak.time or self.cache.time or 0.0

    def _solve(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, *args, ** kwargs):

        # self.constraints.update(constraints)

        logger.debug(f"Solve Equilibrium [{self.__class__.__name__}] at: Start")

        self._solve(*args, ** kwargs)

        logger.debug(f"Solve Equilibrium [{self.__class__.__name__}] at: End")

        self.update_cache()

    def update_cache(self):
        del self.global_quantities
        del self.profiles_1d
        del self.profiles_2d
        del self.boundary
        del self.boundary_separatrix
        del self.flux_surface

        # if isinstance(self._cache, LazyProxy):
        #     self._cache = AttributeTree()

    @cached_property
    def profiles_1d(self):
        return Equilibrium.Profiles1D(self._cache.profiles_1d,
                                      psi_norm=self._psi_norm,
                                      equilibrium=self)

    @cached_property
    def profiles_2d(self):
        return Equilibrium.Profiles2D(self._cache.profiles_2d, equilibrium=self)

    @cached_property
    def global_quantities(self):
        return Equilibrium.GlobalQuantities(equilibrium=self)

    @cached_property
    def boundary(self):
        return Equilibrium.Boundary(equilibrium=self)

    @cached_property
    def boundary_separatrix(self):
        return Equilibrium.BoundarySeparatrix(equilibrium=self)

    @cached_property
    def constraints(self):
        return Equilibrium.Constraints(equilibrium=self)

    @cached_property
    def coordinate_system(self):
        return Equilibrium.CoordinateSystem(equilibrium=self)

    @cached_property
    def flux_surface(self):
        # logger.debug(self.profiles_1d._cache)
        return FluxSurface(self.profiles_2d.psirz,
                           psi_norm=self._psi_norm,
                           ffprime=self.profiles_1d.interpolate("f_df_dpsi"),
                           r0=self.vacuum_toroidal_field.r0, b0=self.vacuum_toroidal_field.b0,
                           coordinate_system=self.coordinate_system,
                           limiter=self.tokamak.wall.limiter_polygon)

    class CoordinateSystem(AttributeTree):
        """Definition of the 2D grid

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

        def __init__(self,  *args, grid=None, equilibrium=None,  grid_type=None, **kwargs):

            super().__init__(*args, **kwargs)
            self.__dict__['_equilibrium'] = equilibrium

            if grid_type is None:
                grid_type = self._equilibrium.cache.coordinate_system.grid_type or 1

            if type(grid_type) is int:
                self.grid_type = AttributeTree(
                    index=grid_type or 1,
                    type="rectangular",
                    description="""Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2). In this case the position
                                arrays should not be filled since they are redundant with grid/dim1 and dim2."""
                )
            else:
                self.grid_type = AttributeTree(grid_type)

            if grid is None:
                self.grid = self._equilibrium.cache.coordinate_system.grid or [32, 128]
            elif isinstance(grid, AttributeTree):
                self.grid = grid
            elif isinstance(grid, list):
                self.grid = AttributeTree(dim1=grid[0], dim2=grid[1])
            else:
                raise ValueError(f"Illegal grid! {grid}")

            if type(self.grid.dim1) is int or np.issubdtype(self.grid.dim1, np.integer):
                self.grid.dim1 = np.linspace(1.0/(self.grid.dim1+1), 1.0,  self.grid.dim1, endpoint=False)

            if type(self.grid.dim2) is int or np.issubdtype(self.grid.dim2, np.integer):
                self.grid.dim2 = np.linspace(
                    0, scipy.constants.pi*2.0,  self.grid.dim2, endpoint=False) + scipy.constants.pi / self.grid.dim2

        @cached_property
        def _metric(self):
            jacobian = np.full(self._shape, np.nan)
            tensor_covariant = np.full([*self._shape, 3, 3], np.nan)
            tensor_contravariant = np.full([*self._shape, 3, 3], np.nan)

            # TODO: not complete

            return AttributeTree({
                "jacobian": jacobian,
                "tensor_covariant": tensor_covariant,
                "tensor_contravariant": tensor_contravariant
            })

        @cached_property
        def r(self):
            """	Values of the major radius on the grid  [m]"""
            return self._mesh.r

        @cached_property
        def z(self):
            """Values of the Height on the grid  [m]"""
            return self._mesh.z

        @cached_property
        def jacobian(self):
            """	Absolute value of the jacobian of the coordinate system  [mixed]"""
            return self._metric.jacobian

        @cached_property
        def tensor_covariant(self):
            """Covariant metric tensor on every point of the grid described by grid_type  [mixed]. """
            return self._metric.tensor_covariant

        @cached_property
        def tensor_contravariant(self):
            """Contravariant metric tensor on every point of the grid described by grid_type  [mixed]"""
            return self._metric.tensor_contravariant

    class Constraints(AttributeTree):
        def __init__(self, *args, equilibrium=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__['_equilibrium'] = equilibrium

    class GlobalQuantities(AttributeTree):
        def __init__(self, cache=None, *args, equilibrium=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__['_equilibrium'] = equilibrium
            self.__dict__['_cache'] = cache

        @property
        def beta_pol(self):
            """Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2]  [-]"""
            return self._equilibrium.cache.global_quantities.beta_pol

        @property
        def beta_tor(self):
            """Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2  [-]"""
            return self._equilibrium.cache.global_quantities.beta_tor

        @property
        def beta_normal(self):
            """Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA]  [-]"""
            return self._equilibrium.cache.global_quantities.beta_normal

        @property
        def ip(self):
            """Plasma current (toroidal component). Positive sign means anti-clockwise when viewed from above.  [A]."""
            return self._equilibrium.cache.global_quantities.ip

        @property
        def li_3(self):
            """Internal inductance  [-]"""
            return self._equilibrium.cache.global_quantities.li

        @property
        def volume(self):
            """Total plasma volume  [m^3]"""
            return self._equilibrium.cache.global_quantities.volume

        @property
        def area(self):
            """Area of the LCFS poloidal cross section  [m^2]"""
            return NotImplemented

        @property
        def surface(self):
            """Surface area of the toroidal flux surface  [m^2]"""
            return NotImplemented

        @property
        def length_pol(self):
            """Poloidal length of the magnetic surface  [m]"""
            return NotImplemented

        # @property
        # def psi_axis(self):
        #     """Poloidal flux at the magnetic axis  [Wb]."""
        #     # return self._equilibrium.cache.global_quantities.psi_axis or
        #     return self._equilibrium.flux_surface.psi_axis

        # @property
        # def psi_boundary(self):
        #     """Poloidal flux at the selected plasma boundary  [Wb]."""
        #     # return self._equilibrium.cache.global_quantities.psi_boundary or
        #     return self._equilibrium.flux_surface.psi_boundary

        # @property
        # def magnetic_axis(self):
        #     """Magnetic axis position and toroidal field	structure"""
        #     return AttributeTree({"r":  self._equilibrium.flux_surface.magnetic_axis.r,
        #                           "z":  self._equilibrium.flux_surface.magnetic_axis.z,
        #                           "b_field_tor": NotImplemented  # self.profiles_2d.b_field_tor(opt[0][0], opt[0][1])
        #                           })

        @cached_property
        def magnetic_axis(self):
            o, _ = self._equilibrium.flux_surface.critical_points
            return o[0]

        @cached_property
        def x_points(self):
            _, x = self._equilibrium.flux_surface.critical_points
            return x

        @cached_property
        def psi_axis(self):
            o, _ = self._equilibrium.flux_surface.critical_points
            return o[0].psi

        @cached_property
        def psi_boundary(self):
            if len(self.x_points) > 0:
                return self.x_points[0].psi
            else:
                raise ValueError(f"No x-point")

        @property
        def q_axis(self):
            """q at the magnetic axis  [-]."""
            return NotImplemented

        @property
        def q_95(self):
            """q at the 95% poloidal flux surface
            (IMAS uses COCOS=11: only positive when toroidal current
            and magnetic field are in same direction)  [-]."""
            return NotImplemented

        @property
        def q_min(self):
            """Minimum q value and position structure"""
            return NotImplemented

        @property
        def energy_mhd(self):
            """Plasma energy content: 3/2 * int(p, dV) with p being the total pressure(thermal + fast particles)[J].  Time-dependent  Scalar [J]"""
            return NotImplemented

    class Profiles1D(Profiles):
        """Equilibrium profiles (1D radial grid) as a function of the poloidal flux	"""

        def __init__(self, cache,   *args,   equilibrium=None, psi_norm=None, ** kwargs):
            if psi_norm is None:
                psi_norm = 129

            if type(psi_norm) is int:
                psi_norm = np.linspace(0, 1.0, psi_norm, endpoint=False)
            super().__init__(cache, *args, x_axis="psi_norm", **kwargs)
            self.__dict__["_equilibrium"] = equilibrium

        @property
        def flux_surface(self):
            return self._equilibrium.flux_surface

        @cached_property
        def psi_norm(self):
            """Normalized poloidal flux  [Wb]. """
            return self._x_axis

        @cached_property
        def psi(self):
            """Poloidal flux  [Wb]. """
            psi_norm = self.psi_norm
            psi_boundary = self._equilibrium.global_quantities.psi_boundary
            psi_axis = self._equilibrium.global_quantities.psi_axis
            return psi_norm*(psi_boundary-psi_axis)+psi_axis

        @cached_property
        def phi(self):
            """Toroidal flux  [Wb] """
            return self._equilibrium.flux_surface.phi

        @cached_property
        def pressure(self):
            """	Pressure  [Pa]"""
            return self.integral("dpressure_dpsi",  self.psi_norm, 1.0)*(self._equilibrium.global_quantities.psi_axis - self._equilibrium.global_quantities.psi_boundary)

        @cached_property
        def pprime(self):
            """Derivative of pressure w.r.t. psi  [Pa.Wb^-1]."""
            return self.dpressure_dpsi

        @cached_property
        def dpressure_dpsi(self):
            """Derivative of pressure w.r.t. psi  [Pa.Wb^-1]."""
            res = super().cache("dpressure_dpsi")
            if not isinstance(res, np.ndarray):
                raise LookupError("dpressure_dpsi")
            return res

        @cached_property
        def ffprime(self):
            """	Derivative of F w.r.t. Psi, multiplied with F  [T^2.m^2/Wb]. """
            return self.f_df_dpsi

        @cached_property
        def f_df_dpsi(self):
            """	Derivative of F w.r.t. Psi, multiplied with F  [T^2.m^2/Wb]. """
            res = self.cache("f_df_dpsi")
            if not isinstance(res, np.ndarray):
                raise LookupError(f"f_df_dpsi")
            return res

        @cached_property
        def f(self):
            """Diamagnetic function (F=R B_Phi)  [T.m]."""
            return self._equilibrium.flux_surface.fpol

        @cached_property
        def fpol(self):
            return self.f

        @cached_property
        def fpol1(self):
            return self._cache.f

        @cached_property
        def j_tor(self):
            """Flux surface averaged toroidal current density = average(j_tor/R) / average(1/R)  [A.m^-2]."""
            return None

        @cached_property
        def j_parallel(self):
            """Flux surface averaged parallel current density = average(j.B) / B0, where B0 = Equilibrium/Global/Toroidal_Field/B0  [A/m^2].  """
            return None

        @cached_property
        def q(self):
            r"""Safety factor
                (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction)  [-].

                .. math:: q(\psi) =\frac{F V^{\prime} \left\langle R^{-2}\right \rangle }{4 \pi^2}
            """
            return self._equilibrium.flux_surface.q

        @cached_property
        def q1(self):
            return self._equilibrium.cache.profiles_1d.q

        @cached_property
        def psi_norm1(self):
            return self.cache("psi_norm")

        @cached_property
        def magnetic_shear(self):
            """Magnetic shear, defined as rho_tor/q . dq/drho_tor[-]	 """
            return self.rho_tor/self.q * self.derivative("q")

        @cached_property
        def r_inboard(self):
            """Radial coordinate(major radius) on the inboard side of the magnetic axis[m]"""
            return None

        @cached_property
        def r_outboard(self):
            """Radial coordinate(major radius) on the outboard side of the magnetic axis[m]"""
            return None

        @cached_property
        def rho_tor(self):
            """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0[m]"""
            return self._equilibrium.flux_surface.rho_tor

        @cached_property
        def rho_tor_norm(self):
            """Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm, is the toroidal flux coordinate at the equilibrium boundary
                (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation)[-]"""
            return self.rho_tor/self.rho_tor[-1]

        @cached_property
        def drho_tor_dpsi(self)	:
            return self._equilibrium.flux_surface.drho_tor_dpsi

        @cached_property
        def dpsi_drho_tor(self)	:
            """Derivative of Psi with respect to Rho_Tor[Wb/m]. """
            return self._equilibrium.flux_surface.dpsi_drho_tor

        @cached_property
        def vprime(self):
            return self.dvolume_dpsi

        @cached_property
        def dvolume_dpsi(self):
            """Radial derivative of the volume enclosed in the flux surface with respect to Psi[m ^ 3.Wb ^ -1]. """
            return self._equilibrium.flux_surface.dvolume_dpsi

        @cached_property
        def volume(self):
            """Volume enclosed in the flux surface[m ^ 3]"""
            return self._equilibrium.flux_surface.volume

        @cached_property
        def dvolume_drho_tor(self)	:
            """Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor[m ^ 2]"""
            return self.dvolume_dpsi * self.dpsi_drho_tor

        @cached_property
        def rho_volume_norm(self)	:
            """Normalised square root of enclosed volume(radial coordinate). The normalizing value is the enclosed volume at the equilibrium boundary
                (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation)[-]"""
            return None

        @cached_property
        def area(self):
            """Cross-sectional area of the flux surface[m ^ 2]"""
            return None

        @cached_property
        def darea_dpsi(self):
            """Radial derivative of the cross-sectional area of the flux surface with respect to psi[m ^ 2.Wb ^ -1]. """
            return None

        @cached_property
        def darea_drho_tor(self)	:
            """Radial derivative of the cross-sectional area of the flux surface with respect to rho_tor[m]"""
            return None

        @cached_property
        def surface(self):
            """Surface area of the toroidal flux surface[m ^ 2]"""
            return None

        @cached_property
        def trapped_fraction(self)	:
            """Trapped particle fraction[-]"""
            return None

        @cached_property
        def b_field_max(self):
            """Maximum(modulus(B)) on the flux surface(always positive, irrespective of the sign convention for the B-field direction)[T]"""
            return None

        @cached_property
        def beta_pol(self):
            """Poloidal beta profile. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip ^ 2][-]"""
            return None

        @cached_property
        def mass_density(self):
            """Mass density[kg.m ^ -3]"""
            return None

        @cached_property
        def gm1(self):
            """Flux surface averaged 1/R ^ 2  [m ^ -2]  """
            return self.flux_surface.gm1

        @cached_property
        def gm2(self):
            r"""Flux surface averaged .. math:: \left | \nabla \rho_{tor}\right|^2/R^2  [m^-2] """
            return self.flux_surface.gm2

        @cached_property
        def gm3(self):
            r"""Flux surface averaged .. math:: \left | \nabla \rho_{tor}\right|^2  [-]	"""
            return self.flux_surface.gm3

        @cached_property
        def gm4(self):
            """Flux surface averaged 1/B ^ 2  [T ^ -2]	"""
            return self.flux_surface.gm4

        @cached_property
        def gm5(self):
            """Flux surface averaged B ^ 2  [T ^ 2]	"""
            return self.flux_surface.gm5

        @cached_property
        def gm6(self):
            r"""Flux surface averaged  .. math:: \left | \nabla \rho_{tor}\right|^2/B^2  [T^-2]	"""
            return self.flux_surface.gm6

        @cached_property
        def gm7(self):
            r"""Flux surface averaged .. math: : \left | \nabla \rho_{tor}\right |  [-]	"""
            return self.flux_surface.gm7

        @cached_property
        def gm8(self):
            """Flux surface averaged R[m]	"""
            return self.flux_surface.gm8

        @cached_property
        def gm9(self):
            """Flux surface averaged 1/R[m ^ -1]          """
            return self.flux_surface.gm9

    class Profiles2D(AttributeTree):
        """
            Equilibrium 2D profiles in the poloidal plane.
        """

        def __init__(self, cache, *args, grid=None, equilibrium=None,  ** kwargs):
            super().__init__(*args, **kwargs)
            self._equilibrium = equilibrium

        @cached_property
        def psirz(self):
            psi_value = self._equilibrium.cache.profiles_2d.psi
            if not self.grid_type.index or self.grid_type.index == 1:  # rectangular	1
                """Cylindrical R, Z ala eqdsk(R=dim1, Z=dim2). In this case the position arrays should not be filled
                 since they are redundant with grid/dim1 and dim2."""

                psi_func = RectBivariateSpline(self.grid.dim1[1:-1], self.grid.dim2[1:-1],  psi_value[1:-1, 1:-1])

            elif self.grid_type.index >= 2 and self.grid_type.index < 91:  # inverse
                """Rhopolar_polar 2D polar coordinates(rho=dim1, theta=dim2) with magnetic axis as centre of grid;
                theta and values following the COCOS=11 convention;
                the polar angle is theta=atan2(z-zaxis,r-raxis) """
                psi_func = SmoothBivariateSpline(self.r.ravel(), self.z.ravel(), psi_value.ravel())

            else:
                raise NotImplementedError()

            return lambda *args, _func=psi_func, grid=False, **kwargs: _func(*args, grid=grid, **kwargs)

        @cached_property
        def grid_type(self):
            res = self._equilibrium.cache.profiles_2d.grid_type
            if res is None:
                res = AttributeTree({
                    "name": "rectangular",
                    "index": 1,
                    "description": """Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2). In this case the position
            arrays should not be filled since they are redundant with grid/dim1 and dim2."""})

            return res

        @cached_property
        def grid(self):
            return AttributeTree(
                dim1=self._equilibrium.cache.profiles_2d.grid.dim1,
                dim2=self._equilibrium.cache.profiles_2d.grid.dim2
            )

        @cached_property
        def _rectangular(self):
            if not self.grid_type.index or self.grid_type.index == 1:
                r, z = np.meshgrid(self.grid.dim1, self.grid.dim2, indexing='ij')
                return AttributeTree(r=r, z=z)
            else:
                raise NotImplementedError()

        @cached_property
        def r(self):
            """Values of the major radius on the grid  [m] """
            return self._rectangular.r

        @cached_property
        def z(self):
            """Values of the Height on the grid  [m] """
            return self._rectangular.z

        @cached_property
        def psi(self):
            """Values of the poloidal flux at the grid in the poloidal plane  [Wb]. """
            return self.psirz(self.r, self.z)

        @cached_property
        def theta(self):
            """	Values of the poloidal angle on the grid  [rad] """
            return NotImplementedError()

        @cached_property
        def phi(self):
            """	Toroidal flux  [Wb]"""
            return self.apply_psifunc("phi")
            # return self._equilibrium.profiles_1d.phi(self.psi(self.r, self.z))

        @cached_property
        def j_tor(self):
            """	Toroidal plasma current density  [A.m^-2]"""
            return self.apply_psifunc("j_tor")

        @cached_property
        def j_parallel(self):
            """	Parallel (to magnetic field) plasma current density  [A.m^-2]"""
            return self.apply_psifunc("j_parallel")

        @cached_property
        def b_field_r(self):
            """R component of the poloidal magnetic field  [T]"""
            return self.psirz(self.r, self.z, dx=1)/(self.r*scipy.constants.pi*2.0)

        @cached_property
        def b_field_z(self):
            """Z component of the poloidal magnetic field  [T]"""
            return - self.psirz(self.r, self.z, dy=1)/(self.r*scipy.constants.pi*2.0)

        @cached_property
        def b_field_tor(self):
            """Toroidal component of the magnetic field  [T]"""
            return self.apply_psifunc("fpol")/self.r

        def apply_psifunc(self, func):
            if isinstance(func, str):
                func = self._equilibrium.profiles_1d.interpolate(func)

            NX = self.grid.dim1.shape[0]
            NY = self.grid.dim2.shape[0]

            res = np.full([NX, NY], np.nan)

            for i in range(NX):
                for j in range(NY):
                    res[i, j] = func(self.psirz(self.r[i, j], self.z[i, j]))
            return res

    class Boundary(AttributeTree):
        def __init__(self, *args, equilibrium=None,    ntheta=None, ** kwargs):
            super().__init__(*args, **kwargs)
            self._equilibrium = equilibrium
            self._ntheta = ntheta or 129

        @cached_property
        def type(self):
            """0 (limiter) or 1 (diverted)  """
            return 1

        @cached_property
        def outline(self):
            """RZ outline of the plasma boundary  """

            # r = self._equilibrium.cache.boundary.outline.r
            # z = self._equilibrium.cache.boundary.outline.z

            # if len(r) == 0 or len(z) == 0:
            boundary = np.array([[r, z] for r, z in self._equilibrium.flux_surface.find_by_psinorm(1.0, self._ntheta)])
            r = boundary[:, 0]
            z = boundary[:, 1]

            return AttributeTree(r=r, z=z)

        @cached_property
        def x_point(self):
            res = AttributeTree()
            _, xpt = self._equilibrium.flux_surface.critical_points
            for p in xpt:
                res[_next_] = p
            return res

        @cached_property
        def psi(self):
            """Value of the poloidal flux at which the boundary is taken  [Wb]"""
            return self._equilibrium.flux_surface.psi_boundary

        @cached_property
        def psi_norm(self):
            """Value of the normalised poloidal flux at which the boundary is taken (typically 99.x %),
                the flux being normalised to its value at the separatrix """
            return self.psi*0.99

        @cached_property
        def geometric_axis(self):
            """RZ position of the geometric axis (defined as (Rmin+Rmax) / 2 and (Zmin+Zmax) / 2 of the boundary)"""
            return AttributeTree(
                {
                    "r": (min(self.outline.r)+max(self.outline.r))/2,
                    "z": (min(self.outline.z)+max(self.outline.z))/2
                })

        @cached_property
        def minor_radius(self):
            """Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary) [m]	"""
            return (max(self.outline.r)-min(self.outline.r))*0.5

        @cached_property
        def elongation(self):
            """Elongation of the plasma boundary Click here for further documentation. [-]	"""
            return (max(self.outline.z)-min(self.outline.z))/(max(self.outline.r)-min(self.outline.r))

        @cached_property
        def elongation_upper(self):
            """Elongation(upper half w.r.t. geometric axis) of the plasma boundary Click here for further documentation. [-]	"""
            return (max(self.outline.z)-self.geometric_axis.z)/(max(self.outline.r)-min(self.outline.r))

        @cached_property
        def elongation_lower(self):
            """Elongation(lower half w.r.t. geometric axis) of the plasma boundary Click here for further documentation. [-]	"""
            return (self.geometric_axis.z-min(self.outline.z))/(max(self.outline.r)-min(self.outline.r))

        @cached_property
        def triangularity(self):
            """Triangularity of the plasma boundary Click here for further documentation. [-]	"""
            return (self.outline.r[np.argmax(self.outline.z)]-self.outline.r[np.argmin(self.outline.z)])/self.minor_radius

        @cached_property
        def triangularity_upper(self):
            """Upper triangularity of the plasma boundary Click here for further documentation. [-]	"""
            return (self.geometric_axis.r - self.outline.r[np.argmax(self.outline.z)])/self.minor_radius

        @cached_property
        def triangularity_lower(self):
            """Lower triangularity of the plasma boundary Click here for further documentation. [-]"""
            return (self.geometric_axis.r - self.outline.r[np.argmin(self.outline.z)])/self.minor_radius

        @cached_property
        def strike_point(self)	:
            """Array of strike points, for each of them the RZ position is given	struct_array [max_size=unbounded]	1- 1...N"""
            return NotImplemented

        @cached_property
        def active_limiter_point(self):
            """	RZ position of the active limiter point (point of the plasma boundary in contact with the limiter)"""
            return NotImplemented

    class BoundarySeparatrix(AttributeTree):
        def __init__(self, *args, equilibrium=None, ** kwargs):
            super().__init__(*args, **kwargs)
            self._equilibrium = equilibrium

    ####################################################################################
    # Plot proflies
    def plot_machine(self, axis=None, *args, coils=True, wall=True, **kwargs):
        if axis is None:
            axis = plt.gca()
        if wall:
            self.tokamak.wall.plot(axis, **kwargs.get("wall", {}))
        if coils:
            self.tokamak.pf_active.plot(axis, **kwargs.get("pf_active", {}))
        axis.axis("scaled")

    def plot_profiles2d(self, axis=None, *args, profiles=[], vec_field=[], boundary=True, levels=32, oxpoints=True,   **kwargs):
        """learn from freegs
        """
        if axis is None:
            axis = plt.gca()

        R = self.profiles_2d.r
        Z = self.profiles_2d.z
        psi = self.profiles_2d.psi

        # psi = (psi - self.global_quantities.psi_axis) / \
        #     (self.global_quantities.psi_boundary - self.global_quantities.psi_axis)

        # if type(levels) is int:
        #     levels = np.linspace(-2, 2,  levels)

        axis.contour(R[1:-1, 1:-1], Z[1:-1, 1:-1], psi[1:-1, 1:-1], levels=levels, linewidths=0.2)

        if boundary:
            boundary_points = np.array([self.boundary.outline.r,
                                        self.boundary.outline.z]).transpose([1, 0])

            axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='dashed',
                                       linewidth=0.5, fill=False, closed=True))
            axis.plot([], [], 'r--', label="Separatrix")

        for idx, p in enumerate(self.boundary.x_point):
            axis.plot(p.r, p.z, 'rx')
            axis.text(p.r, p.z, idx)

        axis.plot([], [], 'rx', label="X-Point")

        axis.plot(self.global_quantities.magnetic_axis.r,
                  self.global_quantities.magnetic_axis.z, 'g.', label="Magnetic axis")

        for k, opts in profiles:
            d = self.profiles_2d[k]
            if d is not NotImplemented and d is not None:
                axis.contourf(R[1:-1, 1:-1], Z[1:-1, 1:-1], d[1:-1, 1:-1], **opts)
        for u, v, opts in vec_field:
            uf = self.profiles_2d[u]
            vf = self.profiles_2d[v]
            axis.streamplot(self.profiles_2d.grid.dim1[1:-1],
                            self.profiles_2d.grid.dim2[1:-1],

                            vf[1:-1, 1:-1].transpose(1, 0),
                            uf[1:-1, 1:-1].transpose(1, 0), **opts)

        axis.set_xlabel(r"Major radius $R$ [m]")
        axis.set_ylabel(r"Height $Z$ [m]")
        axis.set_aspect('equal')

        return axis

    def fetch_profile(self, d):
        if isinstance(d, str):
            data = d
            opts = {"label": d}
        elif isinstance(d, collections.abc.Mapping):
            data = d.get("name", None)
            opts = d.get("opts", {})
        elif isinstance(d, tuple):
            data, opts = d
        elif isinstance(d, AttributeTree):
            data = d.data
            opts = d.opts
        else:
            raise TypeError(f"Illegal profile type! {d}")

        if isinstance(opts, str):
            opts = {"label": opts}

        if isinstance(data, str):
            nlist = data.split(".")
            if len(nlist) == 1:
                data = self.profiles_1d[nlist[0]]
            elif nlist[0] == 'cache':
                data = self.cache.profiles_1d[nlist[1:]]
            else:
                data = self.profiles_1d[nlist]
        elif isinstance(data, list):
            data = np.array(data)
        elif isinstance(d, np.ndarray):
            pass
        else:
            raise TypeError(f"Illegal data type! {type(data)}")

        return data, opts

    def plot_profiles(self, axis, x_axis, profiles):
        if not isinstance(profiles, list):
            profiles = [profiles]

        for idx, data in enumerate(profiles):
            ylabel = None
            opts = {}
            if isinstance(data, tuple):
                data, ylabel = data
            if isinstance(data, str):
                ylabel = data

            if not isinstance(data, list):
                data = [data]

            for d in data:
                value, opts = self.fetch_profile(d)

                if value is not NotImplemented and value is not None and len(value) > 0:
                    axis[idx].plot(x_axis.data, value, **opts)
                else:
                    logger.error(f"Can not find profile '{d}'")

            axis[idx].legend(fontsize=6)

            if ylabel:
                axis[idx].set_ylabel(ylabel, fontsize=6).set_rotation(0)
            axis[idx].labelsize = "media"
            axis[idx].tick_params(labelsize=6)
        return axis[-1]

    def plot(self, *args,
             x_axis=("psi_norm",   r'$(\psi-\psi_{axis})/(\psi_{boundary}-\psi_{axis}) [-]$'),
             profiles=None,
             profiles_2d=[],
             vec_field=[],
             surface_mesh=False,
             **kwargs):

        x_axis, x_axis_opts = self.fetch_profile(x_axis)

        assert (x_axis.data is not NotImplemented)

        if profiles is None or len(profiles) == 0:
            fig, ax_right = plt.subplots(ncols=1, nrows=1, sharex=True)
        else:
            fig, axs = plt.subplots(ncols=2, nrows=len(profiles), sharex=True)
            # left
            ax_left = self.plot_profiles(axs[:, 0], x_axis, profiles)

            ax_left.set_xlabel(x_axis_opts.get("label", "[-]"), fontsize=6)

            # right
            gs = axs[0, 1].get_gridspec()
            for ax in axs[:, 1]:
                ax.remove()  # remove the underlying axes
            ax_right = fig.add_subplot(gs[:, 1])

        if surface_mesh:
            self.flux_surface.plot(ax_right)
        self.plot_profiles2d(ax_right, profiles=profiles_2d, vec_field=vec_field, **kwargs.get("equilibrium", {}))
        self.plot_machine(ax_right, **kwargs.get("machine", {}))

        ax_right.legend()
        fig.tight_layout()

        fig.subplots_adjust(hspace=0)
        fig.align_ylabels()

        return fig

    # # Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2]  [-]
    # self.global_quantities.beta_pol = NotImplemented
    # # Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2  [-]
    # self.global_quantities.beta_tor = NotImplemented
    # # Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA]  [-]
    # self.global_quantities.beta_normal = NotImplemented
    # # Plasma current (toroidal component). Positive sign means anti-clockwise when viewed from above.  [A].
    # self.global_quantities.ip = NotImplemented
    # # Internal inductance  [-]
    # self.global_quantities.li_3 = NotImplemented
    # # Total plasma volume  [m^3]
    # self.global_quantities.volume = NotImplemented
    # # Area of the LCFS poloidal cross section  [m^2]
    # self.global_quantities.area = NotImplemented
    # # Surface area of the toroidal flux surface  [m^2]
    # self.global_quantities.surface = NotImplemented
    # # Poloidal length of the magnetic surface  [m]
    # self.global_quantities.length_pol = NotImplemented
    # # Poloidal flux at the magnetic axis  [Wb].
    # self.global_quantities.psi_axis = NotImplemented
    # # Poloidal flux at the selected plasma boundary  [Wb].
    # self.global_quantities.psi_boundary = NotImplemented
    # # Magnetic axis position and toroidal field	structure
    # self.global_quantities.magnetic_axis = NotImplemented
    # # q at the magnetic axis  [-].
    # self.global_quantities.q_axis = NotImplemented
    # # q at the 95% poloidal flux surface (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction)  [-].
    # self.global_quantities.q_95 = NotImplemented
    # # Minimum q value and position	structure
    # self.global_quantities.q_min = NotImplemented
    # # Plasma energy content = 3/2 * int(p,dV) with p being the total pressure (thermal + fast particles) [J]. Time-dependent; Scalar  [J]
    # self.global_quantities.energy_mhd = NotImplemented
