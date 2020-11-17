
import collections
import functools
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import arctan2, cos, sin, sqrt
from scipy.optimize import root_scalar
from spdm.data.Entry import open_entry
from spdm.util.AttributeTree import AttributeTree, _next_
from spdm.util.Interpolate import (Interpolate1D, Interpolate2D, derivate,
                                   find_critical, find_root, integral,
                                   interpolate)

from spdm.util.utilities import first_not_empty
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module
from sympy import Point, Polygon

from .PFActive import PFActive
from .Wall import Wall

#  psi phi pressure f dpressure_dpsi f_df_dpsi j_parallel q magnetic_shear r_inboard r_outboard rho_tor rho_tor_norm dpsi_drho_tor geometric_axis elongation triangularity_upper triangularity_lower volume rho_volume_norm dvolume_dpsi dvolume_drho_tor area darea_dpsi surface trapped_fraction gm1 gm2 gm3 gm4 gm5 gm6 gm7 gm8 gm9 b_field_max beta_pol mass_density


def first_not_empty(*args):
    return next(x for x in args if len(x) > 0)


class Equilibrium(AttributeTree):
    """
        Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.
        imas dd version 3.28
        ids=equilibrium

        coordinate system
        @ref: O. Sauter and S. Yu Medvedev, "Tokamak coordinate conventions: COCOS", Computer Physics Communications 184, 2 (2013), pp. 293--302.

        COCOS  11

        #################################################################################
        # Top view
        #          ***************
        #         *               *
        #        *   ***********   *
        #       *   *           *   *
        #      *   *             *   *
        #      *   *             *   *
        #  Ip  v   *             *   ^  \\phi
        #      *   *    Z o--->R *   *
        #      *   *             *   *
        #      *   *             *   *
        #      *   *     Bpol    *   *
        #       *   *     o     *   *
        #        *   ***********   *
        #         *               *
        #          ***************
        #            Bpol x
        #
        # Poloidal view
        #     ^Z
        #     |
        #     |       ************
        #     |      *            *
        #     |     *         ^    *
        #     |     *  \\rho /     *
        #     |     *       /      *
        #     +-----*------X-------*---->R
        #     |     *  Ip, \\phi   *
        #     |     *              *
        #     |      *            *
        #     |       *****<******
        #     |       Bpol,\\theta
        #     |
        #
        # Cylindrical coordinate      : (R,\\phi,Z)
        # Poloidal plane coordinate   : (\\rho,\\theta,\\phi)
        #################################################################################
    """
    TOLERANCE = 1.0e-6

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

    def __init__(self,   config,  *args,  tokamak=None,  **kwargs):
        super().__init__(*args, **kwargs)
        self.tokamak = tokamak
        self.load(config)
        # ----------------------------------------------------------
        self.__dict__["_grid_box"] = AttributeTree()

        lim_r = self.tokamak.wall.limiter.outline.r
        lim_z = self.tokamak.wall.limiter.outline.z

        rmin = config.rmin or kwargs.get("rmin", None) or min(lim_r)
        rmax = config.rmax or kwargs.get("rmax", None) or max(lim_r)
        zmin = config.zmin or kwargs.get("zmin", None) or min(lim_z)
        zmax = config.zmax or kwargs.get("zmax", None) or max(lim_z)
        nr = config.nr or kwargs.get("nr", None) or 129
        nz = config.nz or kwargs.get("nz", None) or 129
        self._grid_box.dim1 = np.linspace(rmin-0.1*(rmax-rmin), rmax+0.1*(rmax-rmin), nr)
        self._grid_box.dim2 = np.linspace(zmin-0.1*(zmax-zmin), zmax+0.1*(zmax-zmin), nz)
        # ----------------------------------------------------------
        npsi = config.npsi or kwargs.get("npsi", None) or 16
        ntheta = config.ntheta or kwargs.get("ntheta", None) or 64

        self.__dict__["_time"] = 0.0

    def load(self, config):
        self._cache = config

    @cached_property
    def cache(self):
        if isinstance(self._cache, LazyProxy):
            res = self._cache()
        elif isinstance(self._cache, AttributeTree):
            res = self._cache
        else:
            res = AttributeTree(self._cache)

        return res

    def _solve(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, *args, time=0.0,  ** kwargs):

        # self.constraints.update(constraints)

        logger.debug(f"Solve Equilibrium [{self.__class__.__name__}] at time={self.time} : Start")

        self._solve(*args, time=time,  ** kwargs)

        logger.debug(f"Solve Equilibrium [{self.__class__.__name__}] at time={self.time} : End")

        # change time to invalid all cache
        self.time = time
        # self.clear_cache()

    def clear_cache(self):
        del self.global_quantities
        del self.profiles_1d
        del self.profiles_2d
        del self.boundary
        del self.boundary_separatrix
        del self.critical_points

        raise NotImplementedError()

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        """ Change time will invalid all cache,  """
        self._time = t
        self.clear_cache()

    @cached_property
    def vacuum_toroidal_field(self):
        return AttributeTree(
            r0=self.tokamak.vacuum_toroidal_field.r0,
            b0=self.tokamak.vacuum_toroidal_field.b0
        )

    @cached_property
    def profiles_1d(self):
        return Equilibrium.Profiles1D(self)

    @cached_property
    def profiles_2d(self):
        return Equilibrium.Profiles2D(self)

    @cached_property
    def global_quantities(self):
        return Equilibrium.GlobalQuantities(self)

    @cached_property
    def boundary(self):
        return Equilibrium.Boundary(self)

    @cached_property
    def boundary_separatrix(self):
        return Equilibrium.BoundarySeparatrix(self)

    @cached_property
    def constraints(self):
        return Equilibrium.Constraints(self)

    @cached_property
    def coordinate_system(self):
        return Equilibrium.CoordinateSystem(self)

    @cached_property
    def critical_points(self):
        psi = Interpolate2D(self.profiles_2d.grid.dim1[1:-1],
                            self.profiles_2d.grid.dim2[1:-1],
                            self.profiles_2d.psi[1:-1, 1:-1])

        limiter_points = np.array([self.tokamak.wall.limiter.outline.r,
                                   self.tokamak.wall.limiter.outline.z]).transpose([1, 0])
        limiter_polygon = Polygon(*map(Point, limiter_points))

        opoints = []
        xpoints = []
        for r, z, tag in find_critical(psi):

            # Remove points outside the vacuum wall
            if not limiter_polygon.encloses(Point(r, z)):
                continue

            if tag < 0.0:  # saddle/X-point
                xpoints.append((r, z, psi(r, z)))
            else:  # extremum/ O-point
                opoints.append((r, z, psi(r, z)))

        if not opoints:
            raise RuntimeError(f"Can not find o-point!")
        else:
            Rmid = 0.5*(self.profiles_2d.grid.dim1[-1] + self.profiles_2d.grid.dim1[0])
            Zmid = 0.5*(self.profiles_2d.grid.dim2[-1] + self.profiles_2d.grid.dim2[0])
            opoints.sort(key=lambda x: (x[0] - Rmid)**2 + (x[1] - Zmid)**2)
            psi_axis = opoints[0][2]

            xpoints.sort(key=lambda x: (x[2] - psi_axis)**2)

        return opoints, xpoints

    def _find_psi(self, psi_func, psival, r0, z0, r1, z1):
        if np.isclose(psi_func(r1, z1), psival):
            return r1, z1

        try:
            sol = root_scalar(lambda r: psi_func((1.0-r)*r0+r*r1, (1.0-r)*z0+r*z1) - psival,
                              bracket=[0, 1], method='brentq')
        except ValueError as error:
            raise ValueError(f"Find root fialed! {error}")

        if not sol.converged:
            raise ValueError(f"Find root fialed!")

        r = sol.root

        return (1.0-r)*r0+r*r1, (1.0-r)*z0+r*z1

    def find_surface(self, psival, theta_dim=None, psi_func=None):

        if theta_dim is None or theta_dim is NotImplemented or len(theta_dim) == 0:
            theta_dim = self.coordinate_system.grid.dim2
        elif type(theta_dim) is not int:
            theta_dim = np.linspace(0, scipy.constants.pi*2.0, theta_dim)
        else:
            theta_dim = theta_dim

        if psi_func is None:
            psi_func = Interpolate2D(self.profiles_2d.grid.dim1, self.profiles_2d.grid.dim2, self.profiles_2d.psi)

        psival = psival*(self.global_quantities.psi_boundary -
                         self.global_quantities.psi_axis)+self.global_quantities.psi_axis

        r0 = self.global_quantities.magnetic_axis.r
        z0 = self.global_quantities.magnetic_axis.z
        r1 = self.boundary.x_point[0]["r"]
        z1 = self.boundary.x_point[0]["z"]

        theta0 = arctan2(r1 - r0, z1 - z0)  # + scipy.constants.pi/npoints  # avoid x-point
        R = sqrt((r1-r0)**2+(z1-z0)**2)

        for theta in theta_dim+theta0:
            try:
                r, z = self._find_psi(psi_func, psival, r0, z0,  r0 + R * sin(theta), z0 + R * cos(theta))
            except ValueError as error:
                logger.debug(error)
                continue
            else:
                yield r, z
            # finally:
            #     yield r0 + R * sin(theta), z0 + R * cos(theta)

    @cached_property
    def _surface_average(self):
        logger.debug(f"Calculate magnetic flux surface averge")
        assert(self.coordinate_system.grid is not NotImplemented and self.coordinate_system.grid is not None)

        psi = Interpolate2D(self.profiles_2d.grid.dim1[1:-1],
                            self.profiles_2d.grid.dim2[1:-1],
                            self.profiles_2d.psi[1:-1, 1:-1])

        dim_psi = self.coordinate_system.grid.dim1
        dim_theta = self.coordinate_system.grid.dim1

        R = self.coordinate_system.r
        Z = self.coordinate_system.z

        dpsi_dr = psi(R, Z, dx=1)
        dpsi_dz = psi(R, Z, dy=1)
        dR = (np.roll(R, 1, axis=1) - np.roll(R, -1, axis=1))/2.0
        dZ = (np.roll(Z, 1, axis=1) - np.roll(Z, -1, axis=1))/2.0

        # V'(psi)= 2 *pi* int( dl * R / |grad psi|)
        # <a(psi)> =2 *pi* int(a dl * R / |grad psi|)/V'

        #  |grad psi|
        grad_psi_s = sqrt(dpsi_dr**2 + dpsi_dz**2)

        # dl
        dl = sqrt(dR ** 2 + dZ ** 2)

        # fpol = self.profiles_1d.f
        fpol = self.profiles_1d.fpol
        # B= fpol grad phi  + grad psi \time grad phi
        B2 = dpsi_dr**2 + dpsi_dz**2  # + fpol ** 2

        # Vprime =  2 *pi* int( R / |grad psi| * dl )
        Vprime = (2*scipy.constants.pi) * np.sum(R/grad_psi_s*dl, axis=1)

        # <R^-2> =  int(R^-2 * dl * R / |grad psi|)/Vprime
        gm1 = (2*scipy.constants.pi) * np.sum(dl/grad_psi_s/R, axis=1) / Vprime

        # q(psi) = F Vprime <R^-2> /(4 pi**2)
        q = fpol*Vprime * gm1 / (4*scipy.constants.pi**2)

        # phi= int (q)
        # phi = [Interpolate1D(psi_norm, q, p) for p in psi_norm]
        phi = dim_psi  # FIXME

        drho_tor_dpsi = q*sqrt(scipy.constants.pi/(phi*self.vacuum_toroidal_field.b0))

        # <(grad rho)**2>
        gm3 = (2*scipy.constants.pi) * np.sum(grad_psi_s*R*dl, axis=1) / Vprime

        # <(grad rho)**2/B**2>
        gm6 = (2*scipy.constants.pi) * drho_tor_dpsi * np.sum(R*dl/B2, axis=1) / Vprime

        # <R>
        gm8 = (2*scipy.constants.pi) * np.sum(R*R*dl, axis=1) / drho_tor_dpsi / Vprime

        logger.debug(f"Calculate magnetic flux surface averge: Done")

        return AttributeTree({
            "vprime": Vprime,
            "q": q,
            "phi": phi,
            "drho_tor_dpsi": drho_tor_dpsi,
            "gm1": gm1,
            "gm3": gm3,
            "gm6": gm6,
            "gm8": gm8
        })

    class CoordinateSystem(AttributeTree):
        """
            Definition of the 2D grid

            grid.dim1           :   First dimension values {dynamic} [mixed]
            grid.dim2           :   Second dimension values {dynamic} [mixed]
            grid.volume_element :   Elementary plasma volume of plasma enclosed in the cell formed by the nodes
                                [dim1(i) dim2(j)], [dim1(i+1) dim2(j)], [dim1(i) dim2(j+1)]
                                and [dim1(i+1) dim2(j+1)] {dynamic} [m^3]

            grid_type.name      :
            grid_type.index     : 0xFF (rho theta)
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

        def __init__(self, eq,  *args, grid=None, grid_type=None, **kwargs):

            super().__init__(*args, **kwargs)
            self.__dict__['_eq'] = eq

            grid_type = grid_type or self._eq.cache.coordinate_system.grid_type
            self.grid_type = {}
            if not grid_type:
                self.grid_type.index = 0
                self.grid_type.name = ""
                self.grid_type.description = ""
            else:
                self.grid_type = grid_type

            self.grid = grid or self._eq.cache.coordinate_system.grid or {"dim1": 31, "dim2": 128}

            if np.issubdtype(self.grid.dim1, np.integer):
                self.grid.dim1 = np.linspace(0, 1.0, self.grid.dim1)

            if np.issubdtype(self.grid.dim2, np.integer):
                self.grid.dim2 = np.linspace(0, scipy.constants.pi*2.0, self.grid.dim2,
                                             endpoint=False) + scipy.constants.pi/self.grid.dim2

        @cached_property
        def _mesh(self):
            psi = Interpolate2D(self._eq.profiles_2d.grid.dim1[1:-1],
                                self._eq.profiles_2d.grid.dim2[1:-1],
                                self._eq.profiles_2d.psi[1:-1, 1:-1])

            R, Z = np.meshgrid(self.grid.dim1, self.grid.dim2, indexing="ij")

            R0 = self._eq.global_quantities.magnetic_axis.r
            Z0 = self._eq.global_quantities.magnetic_axis.z
            R1 = self._eq.boundary.x_point[0]["r"]
            Z1 = self._eq.boundary.x_point[0]["z"]

            theta0 = arctan2(R1 - R0, Z1 - Z0)  # + scipy.constants.pi/npoints  # avoid x-point

            Rm = sqrt((R1-R0)**2+(Z1-Z0)**2)

            for i, r in enumerate(self.grid.dim1):
                if abs(r) < 1.0e-6:
                    R[i, :] = R0
                    Z[i, :] = Z0
                    continue
                psival = r*(self._eq.global_quantities.psi_boundary -
                            self._eq.global_quantities.psi_axis)+self._eq.global_quantities.psi_axis
                for j, t in enumerate(self.grid.dim2):
                    theta = t+theta0
                    R[i, j],  Z[i, j] = self._eq._find_psi(psi, psival,
                                                           R0,
                                                           Z0,
                                                           R0 + Rm * sin(theta),
                                                           Z0 + Rm * cos(theta))
            if self.grid_type.index == 0:
                return AttributeTree(r=R,  z=Z)
            else:
                raise ValueError(f"Unknown grid type {self.grid_type}")
                # return self._meshgrid(self.grid_type.index, R, Z, psi)

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
            """	Values of the major radius on the grid {dynamic} [m]"""
            return self._mesh.r

        @cached_property
        def z(self):
            """Values of the Height on the grid {dynamic} [m]"""
            return self._mesh.z

        @cached_property
        def jacobian(self):
            """	Absolute value of the jacobian of the coordinate system {dynamic} [mixed]"""
            return self._metric.jacobian

        @cached_property
        def tensor_covariant(self):
            """Covariant metric tensor on every point of the grid described by grid_type {dynamic} [mixed]. """
            return self._metric.tensor_covariant

        @cached_property
        def tensor_contravariant(self):
            """Contravariant metric tensor on every point of the grid described by grid_type {dynamic} [mixed]"""
            return self._metric.tensor_contravariant

    class Constraints(AttributeTree):
        def __init__(self, eq, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__['_eq'] = eq

    class GlobalQuantities(AttributeTree):
        def __init__(self, eq, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__['_eq'] = eq

        @property
        def beta_pol(self):
            """ Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2] {dynamic} [-]"""
            return self._eq.cache.global_quantities.beta_pol

        @property
        def beta_tor(self):
            """ Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2 {dynamic} [-]"""
            return self._eq.cache.global_quantities.beta_tor

        @property
        def beta_normal(self):
            """ Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA] {dynamic} [-]"""
            return self._eq.cache.global_quantities.beta_normal

        @property
        def ip(self):
            """ Plasma current (toroidal component). Positive sign means anti-clockwise when viewed from above. {dynamic} [A]."""
            return NotImplemented

        @property
        def li_3(self):
            """ Internal inductance {dynamic} [-]"""
            return NotImplemented

        @property
        def volume(self):
            """ Total plasma volume {dynamic} [m^3]"""
            return self._eq.cache.global_quantities.volume

        @property
        def area(self):
            """ Area of the LCFS poloidal cross section {dynamic} [m^2]"""
            return NotImplemented

        @property
        def surface(self):
            """ Surface area of the toroidal flux surface {dynamic} [m^2]"""
            return NotImplemented

        @property
        def length_pol(self):
            """ Poloidal length of the magnetic surface {dynamic} [m]"""
            return NotImplemented

        @property
        def psi_axis(self):
            """ Poloidal flux at the magnetic axis {dynamic} [Wb]."""
            return self._eq.cache.global_quantities.psi_axis or self._eq.critical_points[0][0][2]

        @property
        def psi_boundary(self):
            """ Poloidal flux at the selected plasma boundary {dynamic} [Wb]."""
            return self._eq.cache.global_quantities.psi_boundary or self._eq.critical_points[1][0][2]

        @property
        def magnetic_axis(self):
            """ Magnetic axis position and toroidal field	structure"""
            opt, _ = self._eq.critical_points
            return AttributeTree({"r":  opt[0][0],
                                  "z":  opt[0][1],
                                  "b_field_tor": NotImplemented  # self.profiles_2d.b_field_tor(opt[0][0], opt[0][1])
                                  })

        @property
        def q_axis(self):
            """ q at the magnetic axis {dynamic} [-]."""
            return NotImplemented

        @property
        def q_95(self):
            """ q at the 95% poloidal flux surface
            (IMAS uses COCOS=11: only positive when toroidal current
            and magnetic field are in same direction) {dynamic} [-]."""

            return NotImplemented

        @property
        def q_min(self):
            """ Minimum q value and position	structure"""

            return NotImplemented

        @property
        def energy_mhd(self):
            """ Plasma energy content:
                  3/2 * int(p, dV) with p being the total pressure(thermal + fast particles)[J].
                Time-dependent  Scalar {dynamic}[J]"""
            return NotImplemented

    class Profiles1D(AttributeTree):
        """Equilibrium profiles (1D radial grid) as a function of the poloidal flux	"""

        def __init__(self, eq,  *args, npsi=None,  ** kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__["_eq"] = eq
            self.__dict__["_npsi"] = npsi or 129

        @cached_property
        def psi(self):
            """Poloidal flux {dynamic} [Wb]. """
            res = self._eq.cache.profiles_1d.psi
            if res is not NotImplemented and res is not None and len(res) > 0:
                pass
            elif self.psi_norm is not NotImplemented and self.psi_norm is not None and len(self.psi_norm) > 0:
                res = self.psi_norm * (self._eq.global_quantities.psi_boundary -
                                       self._eq.global_quantities.psi_axis) + self._eq.global_quantities.psi_axis

            return self._eq.cache.profiles_1d.psi

        @cached_property
        def psi_norm(self):
            """Normalized poloidal flux {dynamic} [Wb]. """
            res = self._eq.cache.profiles_1d.psi_norm
            if res is NotImplemented or res is None or len(res) == 0:
                res = np.linspace(0, 1, self._npsi)
            return res

        @cached_property
        def phi(self):
            """Toroidal flux {dynamic} [Wb]."""
            return self._eq._surface_average.phi

        @cached_property
        def pressure(self):
            """	Pressure {dynamic} [Pa]"""
            # pprime = self.dpresure_dpsi
            # return [pprime.integral(0, self.psi) for psi in self.psi]
            res = self._eq.cache.profiles_1d.pressure
            if len(res) == 0:
                res = self._eq._surface_average.pressure
            return res

        @cached_property
        def f(self):
            """Diamagnetic function (F=R B_Phi) {dynamic} [T.m]."""
            # ffprime = self.f_df_dpsi
            # f2 = [2*ffprime.integral(0, psi) for psi in self.psi]
            # return np.sqrt(f2)
            res = self._eq.cache.profiles_1d.f

            if len(res) == 0:
                res = self._eq.cache.profiles_1d.fpol

            if res is NotImplemented or res is None or len(res) == 0:
                raise KeyError("f")
            return res

        @cached_property
        def fpol(self):
            return self.f

        @cached_property
        def pprime(self):
            """Derivative of pressure w.r.t. psi {dynamic} [Pa.Wb^-1]."""
            return self.dpressure_dpsi

        @cached_property
        def dpressure_dpsi(self):
            """Derivative of pressure w.r.t. psi {dynamic} [Pa.Wb^-1]."""
            res = self._eq.cache.profiles_1d.dpressure_dpsi
            if len(res) == 0:
                res = self._eq.cache.profiles_1d.pprime
            if len(res) == 0:
                res = self._eq._surface_average.pprime
            return res

        @cached_property
        def ffprime(self):
            """	Derivative of F w.r.t. Psi, multiplied with F {dynamic} [T^2.m^2/Wb]. """
            return self.f_df_dpsi

        @cached_property
        def f_df_dpsi(self):
            """	Derivative of F w.r.t. Psi, multiplied with F {dynamic} [T^2.m^2/Wb]. """
            res = self._eq.cache.profiles_1d.f_df_dpsi
            if len(res) == 0:
                res = self._eq.cache.profiles_1d.ffprime
            return res

        @cached_property
        def j_tor(self):
            """Flux surface averaged toroidal current density = average(j_tor/R) / average(1/R) {dynamic} [A.m^-2]."""
            return NotImplemented

        @cached_property
        def j_parallel(self):
            """Flux surface averaged parallel current density = average(j.B) / B0, where B0 = Equilibrium/Global/Toroidal_Field/B0 {dynamic} [A/m^2].  """
            return NotImplemented

        @cached_property
        def q(self):
            """Safety factor (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction) {dynamic} [-]. """
            res = self._eq.cache.profiles_1d.q
            if len(res) == 0:
                res = self._eq._surface_average.q
            return res

        @cached_property
        def magnetic_shear(self):
            """Magnetic shear, defined as rho_tor/q . dq/drho_tor {dynamic} [-]	 """
            return self.rho_tor/self.q * derivate(self.q, self.dpsi_drho_tor)

        @cached_property
        def r_inboard(self):
            """Radial coordinate (major radius) on the inboard side of the magnetic axis {dynamic} [m]"""
            return NotImplemented

        @cached_property
        def r_outboard(self):
            """Radial coordinate (major radius) on the outboard side of the magnetic axis {dynamic} [m]"""
            return NotImplemented

        @cached_property
        def rho_tor(self):
            """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0 {dynamic} [m]"""
            b0 = self._eq.vacuum_toroidal_field.b0
            return np.sqrt(self.phi)/np.sqrt(scipy.constants.pi*b0)

        @cached_property
        def rho_tor_norm(self):
            """Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm, is the toroidal flux coordinate at the equilibrium boundary
                (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation) {dynamic} [-]"""
            return self.rho_tor/self.rho_tor[-1]

        @cached_property
        def dpsi_drho_tor(self)	:
            """Derivative of Psi with respect to Rho_Tor {dynamic} [Wb/m]. """
            return self._eq.vacuum_toroidal_field.b0*self.rho_tor/self.q

        @cached_property
        def volume(self):
            """Volume enclosed in the flux surface {dynamic} [m^3]"""
            vprime = self.dvolume_dpsi
            psi_scale = self._eq.global_quantities.psi_boundary - self._eq.global_quantities.psi_axis
            return [integral(vprime, 0, psi)*psi_scale for psi in self.psi_norm]

        @cached_property
        def rho_volume_norm(self)	:
            """Normalised square root of enclosed volume (radial coordinate). The normalizing value is the enclosed volume at the equilibrium boundary
                (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation) {dynamic} [-]"""
            return NotImplemented

        @cached_property
        def vprime(self):
            return self.dvolume_dpsi

        @cached_property
        def dvolume_dpsi(self):
            """Radial derivative of the volume enclosed in the flux surface with respect to Psi {dynamic} [m^3.Wb^-1]. """
            return self._eq._surface_average.dvolume_dpsi

        @cached_property
        def dvolume_drho_tor(self)	:
            """Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor {dynamic} [m^2]"""
            return self.dvolume_dpsi*self.dpsi_drho_tor

        @cached_property
        def area(self):
            """Cross-sectional area of the flux surface {dynamic} [m^2]"""
            return NotImplemented

        @cached_property
        def darea_dpsi(self):
            """Radial derivative of the cross-sectional area of the flux surface with respect to psi {dynamic} [m^2.Wb^-1]. """
            return NotImplemented

        @cached_property
        def darea_drho_tor(self)	:
            """Radial derivative of the cross-sectional area of the flux surface with respect to rho_tor {dynamic} [m]"""
            return NotImplemented

        @cached_property
        def surface(self):
            """Surface area of the toroidal flux surface {dynamic} [m^2]"""
            return NotImplemented

        @cached_property
        def trapped_fraction(self)	:
            """Trapped particle fraction {dynamic} [-]"""
            return NotImplemented

        @cached_property
        def b_field_max(self):
            """Maximum(modulus(B)) on the flux surface (always positive, irrespective of the sign convention for the B-field direction) {dynamic} [T]"""
            return NotImplemented

        @cached_property
        def beta_pol(self):
            """Poloidal beta profile. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2] {dynamic} [-]"""
            return NotImplemented

        @cached_property
        def mass_density(self):
            """Mass density {dynamic} [kg.m^-3]"""
            return NotImplemented

        @cached_property
        def gm1(self):
            """ Flux surface averaged 1/R^2 {dynamic} [m^-2]  """
            return NotImplemented

        @cached_property
        def gm2(self):
            """ Flux surface averaged |grad_rho_tor|^2/R^2 {dynamic} [m^-2] """
            return NotImplemented

        @cached_property
        def gm3(self):
            """ Flux surface averaged |grad_rho_tor|^2 {dynamic} [-]	"""
            return self._eq._surface_average.gm3

        @cached_property
        def gm4(self):
            """ Flux surface averaged 1/B^2 {dynamic} [T^-2]	"""
            return NotImplemented

        @cached_property
        def gm5(self):
            """ Flux surface averaged B^2 {dynamic} [T^2]	"""
            return NotImplemented

        @cached_property
        def gm6(self):
            """ Flux surface averaged |grad_rho_tor|^2/B^2 {dynamic} [T^-2]	"""
            return self._eq._surface_average.gm6

        @cached_property
        def gm7(self):
            """ Flux surface averaged |grad_rho_tor| {dynamic} [-]	"""
            return NotImplemented

        @cached_property
        def gm8(self):
            """ Flux surface averaged R {dynamic} [m]	"""
            return self._eq._surface_average.gm8

        @cached_property
        def gm9(self):
            """ Flux surface averaged 1/R {dynamic} [m^-1]          """
            return NotImplemented

    class Profiles2D(AttributeTree):
        """
            Equilibrium 2D profiles in the poloidal plane. 
        """

        def __init__(self, eq, *args, grid=None,  ** kwargs):
            super().__init__(*args, **kwargs)
            self._eq = eq

        @cached_property
        def grid_type(self):
            res = self._eq.cache.profiles_2d.grid_type
            if res is NotImplemented or res is None:
                res = AttributeTree({
                    "name": "rectangular",
                    "index": 1,
                    "description": """ Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2). In this case the position
            arrays should not be filled since they are redundant with grid/dim1 and dim2."""})

            return res

        @cached_property
        def grid(self):
            return AttributeTree(
                dim1=self._eq.cache.profiles_2d.grid.dim1,
                dim2=self._eq.cache.profiles_2d.grid.dim2
            )

        @cached_property
        def _rectangular(self):
            if not self.grid_type.index or self.grid_type.index == 1:
                r, z = np.meshgrid(self.grid.dim1, self.grid.dim2, indexing='ij')
                return AttributeTree(r=r, z=z)
            else:
                return AttributeTree(r=NotImplemented, z=NotImplemented)

        @cached_property
        def r(self):
            """Values of the major radius on the grid {dynamic} [m] """
            res = self._eq.cache.profiles_2d.r
            if res is NotImplemented or res is None or len(res) == 0:
                res = self._rectangular.r
            return res

        @cached_property
        def z(self):
            """Values of the Height on the grid {dynamic} [m] """
            res = self._eq.cache.profiles_2d.z
            if res is NotImplemented or res is None or len(res) == 0:
                res = self._rectangular.z
            return res

        @cached_property
        def _rho_polar(self):
            if self.grid_type.index > 1 and self.grid_type.index < 91:
                psi, theta = np.meshgrid(self.grid.dim1, self.grid.dim2, indexing='ij')
                return AttributeTree(psi=psi, thetaz=theta)
            else:
                return AttributeTree(psi=NotImplemented, theta=NotImplemented)

        @cached_property
        def psi(self):
            """Values of the poloidal flux at the grid in the poloidal plane {dynamic} [Wb]. """
            res = self._eq.cache.profiles_2d.psi
            if res is NotImplemented or res is None or len(res) == 0:
                res = self._rho_polar.psi
            return res

        @cached_property
        def theta(self):
            """	Values of the poloidal angle on the grid {dynamic} [rad] """
            res = self._eq.cache.profiles_2d.theta
            if res is NotImplemented or res is None or len(res) == 0:
                res = self._rho_polar.theta

            return res

        @cached_property
        def phi(self):
            """	Toroidal flux {dynamic} [Wb]"""
            return NotImplemented

        @cached_property
        def j_tor(self):
            """	Toroidal plasma current density {dynamic} [A.m^-2]"""
            return NotImplemented

        @cached_property
        def j_parallel(self):
            """	Parallel (to magnetic field) plasma current density {dynamic} [A.m^-2]"""
            return NotImplemented

        @cached_property
        def _b_field(self):
            """Values of the major radius on the grid {dynamic} [m] """
            return AttributeTree(
                r=None,
                z=None,
                tor=None
            )

        @cached_property
        def b_field_r(self):
            """R component of the poloidal magnetic field {dynamic} [T]"""
            res = self._eq.cache.profiles_2d.b_field_r
            if res is NotImplemented or res is None or len(res) == 0:
                res = self._b_field.r
            return res

        @cached_property
        def b_field_z(self):
            """Z component of the poloidal magnetic field {dynamic} [T]"""
            res = self._eq.cache.profiles_2d.b_field_z
            if res is NotImplemented or res is None or len(res) == 0:
                res = self._b_field.z
            return res

        @cached_property
        def b_field_tor(self):
            """Toroidal component of the magnetic field {dynamic} [T]"""
            res = self._eq.cache.profiles_2d.b_field_tor
            if res is not NotImplemented and res is not None and len(res) > 0:
                pass
            else:
                res = self._b_field.tor
            return res

    class Boundary(AttributeTree):
        def __init__(self, eq, *args,   ntheta=None, ** kwargs):
            super().__init__(*args, **kwargs)
            self._eq = eq

        @cached_property
        def type(self):
            """0 (limiter) or 1 (diverted) {dynamic} """
            return 1

        @cached_property
        def outline(self):
            """ RZ outline of the plasma boundary  """

            r = self._eq.cache.boundary.outline.r
            z = self._eq.cache.boundary.outline.z

            if len(r) == 0 or len(z) == 0:
                boundary = np.array([p for p in self._eq.find_surface(1.0)])
                r = boundary[:, 0]
                z = boundary[:, 1]

            return AttributeTree(r=r, z=z)

        @cached_property
        def x_point(self):
            res = AttributeTree()
            _, xpt = self._eq.critical_points
            for r, z, _ in xpt:
                res[_next_] = {"r": r, "z": z}

            return res

        @cached_property
        def psi(self):
            """ Value of the poloidal flux at which the boundary is taken {dynamic} [Wb]"""
            _, xpt = self._eq.critical_points
            return xpt[0][2]

        @cached_property
        def psi_norm(self):
            """ Value of the normalised poloidal flux at which the boundary is taken (typically 99.x %),
                the flux being normalised to its value at the separatrix {dynamic}"""
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
            """Minor radius of the plasma boundary(defined as (Rmax-Rmin) / 2 of the boundary) {dynamic}[m]	FLT_0D"""
            return (max(self.outline.r)-min(self.outline.r))*0.5

        @cached_property
        def elongation(self):
            """Elongation of the plasma boundary Click here for further documentation. {dynamic}[-]	FLT_0D"""
            return (max(self.outline.z)-min(self.outline.z))/(max(self.outline.r)-min(self.outline.r))

        @cached_property
        def elongation_upper(self):
            """Elongation(upper half w.r.t. geometric axis) of the plasma boundary Click here for further documentation. {dynamic}[-]	FLT_0D"""
            return (max(self.outline.z)-self.geometric_axis.z)/(max(self.outline.r)-min(self.outline.r))

        @cached_property
        def elongation_lower(self):
            """Elongation(lower half w.r.t. geometric axis) of the plasma boundary Click here for further documentation. {dynamic}[-]	FLT_0D"""
            return (self.geometric_axis.z-min(self.outline.z))/(max(self.outline.r)-min(self.outline.r))

        @cached_property
        def triangularity(self):
            """Triangularity of the plasma boundary Click here for further documentation. {dynamic}[-]	FLT_0D"""
            return (self.outline.r[np.argmax(self.outline.z)]-self.outline.r[np.argmin(self.outline.z)])/self.minor_radius

        @cached_property
        def triangularity_upper(self):
            """Upper triangularity of the plasma boundary Click here for further documentation. {dynamic}[-]	FLT_0D"""
            return (self.geometric_axis.r - self.outline.r[np.argmax(self.outline.z)])/self.minor_radius

        @cached_property
        def triangularity_lower(self):
            """Lower triangularity of the plasma boundary Click here for further documentation. {dynamic}[-]"""
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
        def __init__(self, eq, *args,  ** kwargs):
            super().__init__(*args, **kwargs)
            self._eq = eq

    ####################################################################################
    # Plot proflies

    def plot(self, axis=None, *args, levels=32, oxpoints=True, **kwargs):
        """ learn from freegs
        """
        if axis is None:
            axis = plt.gca()

        R = self.profiles_2d.r
        Z = self.profiles_2d.z
        psi = self.profiles_2d.psi

        psi = (psi - self.global_quantities.psi_axis) / \
            (self.global_quantities.psi_boundary - self.global_quantities.psi_axis)

        if type(levels) is int:
            levels = np.linspace(-2, 2,  levels)

        axis.contour(R[1:-1, 1:-1], Z[1:-1, 1:-1], psi[1:-1, 1:-1], levels=levels, linewidths=0.2)

        for idx, p in enumerate(self.boundary.x_point):
            axis.plot(p.r, p.z, 'rx')
            axis.text(p.r, p.z, idx)

        boundary_points = np.array([self.boundary.outline.r,
                                    self.boundary.outline.z]).transpose([1, 0])

        axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='dashed',
                                   linewidth=0.5, fill=False, closed=True))
        axis.plot([], [], 'r--', label="Separatrix")

        axis.plot(self.global_quantities.magnetic_axis.r,
                  self.global_quantities.magnetic_axis.z, 'g.', label="Magnetic axis")

        return axis

    def plot_surface_mesh(self, axis):
        axis.plot(self.coordinate_system.r, self.coordinate_system.z, "b--", linewidth=0.1)
        axis.plot(self.coordinate_system.r.transpose(1, 0),
                  self.coordinate_system.z.transpose(1, 0), "b--", linewidth=0.1)
        return axis

    def plot_full(self,  profiles=None, *args,
                  x_axis={"key": "psi_norm", "label": r'$(\psi-\psi_{axis})/(\psi_{boundary}-\psi_{axis})$'},
                  **kwargs):

        if isinstance(profiles, str):
            profiles = profiles.split(",")
        elif profiles is None:
            profiles = [
                ("q", r"q", r"$[-]$"),
                ("pprime", r"$p^{\prime}$", r"$[Pa/Wb]$"),
                ("ffprime", r"$f f^{\prime}$", r"$[T^2 \cdot m^2/Wb]$"),
                ("fpol", r"$f_{pol}$", r"$[T \cdot m]$"),
                # ("pressure", r"pressure", r"$[Pa]$")
            ]

        def fetch_profile(p):
            if isinstance(p, str):
                p = AttributeTree(key=p)
            elif isinstance(p, tuple):
                key, label, unit = p
                p = AttributeTree(key=key,  label=label, unit=unit)
            elif not isinstance(p, AttributeTree):
                p = AttributeTree(p)

            try:
                d = self.profiles_1d[p["key"]]
            except KeyError as error:
                logger.warning(f"Plot profile  failed!  {error}")
                d = None

            if d is None or d is NotImplemented or len(d) == 0:
                pass
            else:
                p.data = d

            return p

        x_axis = fetch_profile(x_axis)

        assert (x_axis.data is not NotImplemented)

        profiles = [p for p in map(fetch_profile, profiles) if len(p.data) > 0]

        nprofiles = len(profiles)

        fig, axs = plt.subplots(ncols=2, nrows=nprofiles, sharex=True)
        gs = axs[0, 1].get_gridspec()
        # remove the underlying axes
        for ax in axs[:, 1]:
            ax.remove()
        ax_right = fig.add_subplot(gs[:, 1])

        self.tokamak.wall.plot(ax_right, **kwargs.get("wall", {}))
        self.tokamak.pf_active.plot(ax_right, **kwargs.get("pf_active", {}))
        self.plot(ax_right, **kwargs.get("equilibrium", {}))

        ax_right.set_aspect('equal')
        ax_right.set_xlabel(r"Major radius $R$ [m]")
        ax_right.set_ylabel(r"Height $Z$ [m]")
        ax_right.legend()

        for idx, p in enumerate(profiles):

            axs[idx, 0].plot(x_axis.data, p.data)
            # if p.unit is not None:
            axs[idx, 0].set_ylabel(f"{p.label or p.key}{p.unit or '[-]'}")
            # axs[idx, 0].legend()

        axs[nprofiles-1, 0].set_xlabel(x_axis.label+(x_axis.unit or ""))

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.align_ylabels()

        return fig

    # # Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2] {dynamic} [-]	FLT_0D
    # self.global_quantities.beta_pol = NotImplemented
    # # Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2 {dynamic} [-]	FLT_0D
    # self.global_quantities.beta_tor = NotImplemented
    # # Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA] {dynamic} [-]	FLT_0D
    # self.global_quantities.beta_normal = NotImplemented
    # # Plasma current (toroidal component). Positive sign means anti-clockwise when viewed from above. {dynamic} [A].
    # self.global_quantities.ip = NotImplemented
    # # Internal inductance {dynamic} [-]	FLT_0D
    # self.global_quantities.li_3 = NotImplemented
    # # Total plasma volume {dynamic} [m^3]	FLT_0D
    # self.global_quantities.volume = NotImplemented
    # # Area of the LCFS poloidal cross section {dynamic} [m^2]	FLT_0D
    # self.global_quantities.area = NotImplemented
    # # Surface area of the toroidal flux surface {dynamic} [m^2]	FLT_0D
    # self.global_quantities.surface = NotImplemented
    # # Poloidal length of the magnetic surface {dynamic} [m]	FLT_0D
    # self.global_quantities.length_pol = NotImplemented
    # # Poloidal flux at the magnetic axis {dynamic} [Wb].
    # self.global_quantities.psi_axis = NotImplemented
    # # Poloidal flux at the selected plasma boundary {dynamic} [Wb].
    # self.global_quantities.psi_boundary = NotImplemented
    # # Magnetic axis position and toroidal field	structure
    # self.global_quantities.magnetic_axis = NotImplemented
    # # q at the magnetic axis {dynamic} [-].
    # self.global_quantities.q_axis = NotImplemented
    # # q at the 95% poloidal flux surface (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction) {dynamic} [-].
    # self.global_quantities.q_95 = NotImplemented
    # # Minimum q value and position	structure
    # self.global_quantities.q_min = NotImplemented
    # # Plasma energy content = 3/2 * int(p,dV) with p being the total pressure (thermal + fast particles) [J]. Time-dependent; Scalar {dynamic} [J]
    # self.global_quantities.energy_mhd = NotImplemented
