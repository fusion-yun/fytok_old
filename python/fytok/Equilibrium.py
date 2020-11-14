
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
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module
from sympy import Point, Polygon

from .PFActive import PFActive
from .Wall import Wall

#  psi phi pressure f dpressure_dpsi f_df_dpsi j_parallel q magnetic_shear r_inboard r_outboard rho_tor rho_tor_norm dpsi_drho_tor geometric_axis elongation triangularity_upper triangularity_lower volume rho_volume_norm dvolume_dpsi dvolume_drho_tor area darea_dpsi surface trapped_fraction gm1 gm2 gm3 gm4 gm5 gm6 gm7 gm8 gm9 b_field_max beta_pol mass_density


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
    @staticmethod
    def __new__(cls,   config,  *args, **kwargs):
        if cls is not Equilibrium:
            return super(Equilibrium, cls).__new__(cls)

        backend = str(config.engine or "") or "FreeGS"

        plugin_name = f"{__package__}.plugins.equilibrium.Plugin{backend}"

        n_cls = sp_find_module(plugin_name, fragment=f"Equilibrium{backend}")

        if n_cls is None:
            raise ModuleNotFoundError(f"Can not find plugin {plugin_name}#Equilibrium{backend}")

        return AttributeTree.__new__(n_cls)

    def __init__(self,   config,  *args,  tokamak=None, nr=129, nz=129, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokamak = tokamak
        lim_r = self.tokamak.wall.limiter.outline.r
        lim_z = self.tokamak.wall.limiter.outline.z
        rmin = min(lim_r)
        rmax = max(lim_r)
        zmin = min(lim_z)
        zmax = max(lim_z)
        self.coordinate_system.grid.dim1 = np.linspace(rmin-0.1*(rmax-rmin), rmax+0.1*(rmax-rmin), nr)
        self.coordinate_system.grid.dim2 = np.linspace(zmin-0.1*(zmax-zmin), zmax+0.1*(zmax-zmin), nz)
        self.constraints = Equilibrium.Constraints(self)

        self.vacuum_toroidal_field.r0 = self.tokamak.vacuum_toroidal_field.r0
        self.vacuum_toroidal_field.b0 = 1.0

        self.__dict__["_time"] = 0.0

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
    def profiles_1d(self):
        return Equilibrium.Profiles1D(self)

    @cached_property
    def profiles_2d(self):
        return Equilibrium.Profiles2D(self)

    @cached_property
    def critical_points(self):

        R = self.profiles_2d.r
        Z = self.profiles_2d.z
        psi = self.profiles_2d.psi

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

        Rmid = 0.5*(R[-1, 0] + R[0, 0])
        Zmid = 0.5*(Z[0, -1] + Z[0, 0])
        opoints.sort(key=lambda x: (x[0] - Rmid)**2 + (x[1] - Zmid)**2)
        psi_axis = opoints[0][2]
        xpoints.sort(key=lambda x: (x[2] - psi_axis)**2)

        return opoints, xpoints

    @cached_property
    def global_quantities(self):
        return Equilibrium.GlobalQuantities(self)

    @cached_property
    def boundary(self):
        return Equilibrium.Boundary(self)

    @cached_property
    def boundary_separatrix(self):
        return Equilibrium.BoundarySeparatrix(self)

    def _find_psi(self, psival, r0, z0, r1, z1):
        if np.isclose(self.profiles_2d.psi(r1, z1), psival):
            return r1, z1

        try:
            sol = root_scalar(lambda r: self.profiles_2d.psi_norm((1.0-r)*r0+r*z0, (1.0-r)*r1+r*z1) - psival,
                              bracket=[0, 1], method='brentq')
        except ValueError:
            raise ValueError(f"Find root fialed!")

        if not sol.converged:
            raise ValueError(f"Find root fialed!")

        r = sol.root

        return (1.0-r)*r0+r*r1, (1.0-r)*z0+r*z1

    def find_surface(self, psival, npoints=128):

        if not self.boundary.x_point:
            self.update_boundary()

        psi_norm = self.profiles_2d.psi_norm()

        r0 = self.global_quantities.magnetic_axis.r
        z0 = self.global_quantities.magnetic_axis.z
        r1 = self.boundary.x_point[0]["r"]
        z1 = self.boundary.x_point[0]["z"]

        # logger.debug(psival)
        theta0 = arctan2(r1 - r0, z1 - z0)  # + scipy.constants.pi/npoints  # avoid x-point
        R = sqrt((r1-r0)**2+(z1-z0)**2)

        # def f(r, x0, x1, val):
        #     return psi_norm((1.0-r)*x0[0]+r*x1[0], (1.0-r)*x0[1]+r*x1[1]) - val

        for theta in np.linspace(0, scipy.constants.pi*2.0, npoints)+theta0:
            try:
                r, z = self._find_psi(psival, r0, z0,  r0 + R * sin(theta), z0 + R * cos(theta))
            except ValueError:
                continue
            else:
                yield r, z
            # if np.isclose(psi_norm(r1, z1), psival):
            #     yield r1, z1
            #     continue
            # try:
            #     sol = root_scalar(f, bracket=[0, 1], args=([r0, z0], [r1, z1], psival), method='brentq')
            # except ValueError:
            #     continue
            # if not sol.converged:
            #     continue
            # r = sol.root
            # yield (1.0-r)*r0+r*r1, (1.0-r)*z0+r*z1

    class Constraints(AttributeTree):
        def __init__(self, eq, *args, **kwargs):
            super().__init__(eq, *args, **kwargs)
            self.__dict__['_eq'] = eq

    class GlobalQuantities(AttributeTree):

        @staticmethod
        def __new__(cls,   eq,  *args, **kwargs):
            if cls is not Equilibrium.GlobalQuantities:
                return super(Equilibrium.GlobalQuantities, cls).__new__(cls)
            ncls = getattr(eq.__class__, cls.__name__, cls)
            return AttributeTree.__new__(ncls)

        def __init__(self, eq, *args, **kwargs):
            super().__init__(eq, *args, **kwargs)
            self.__dict__['_eq'] = eq

        @property
        def beta_pol(self):
            """ Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2] {dynamic} [-]"""
            return NotImplemented

        @property
        def beta_tor(self):
            """ Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2 {dynamic} [-]"""
            return NotImplemented

        @property
        def beta_normal(self):
            """ Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA] {dynamic} [-]"""

            return NotImplemented

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
            return self._eq._backend.plasmaVolume()

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
            opt, _ = self._eq.critical_points
            return opt[0][2]

        @property
        def psi_boundary(self):
            """ Poloidal flux at the selected plasma boundary {dynamic} [Wb]."""
            _, xpt = self._eq.critical_points
            return xpt[0][2]

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
        @staticmethod
        def __new__(cls,   eq,  *args, **kwargs):
            if cls is not Equilibrium.Profiles1D:
                return super(Equilibrium.Profiles1D, cls).__new__(cls)
            ncls = getattr(eq.__class__, cls.__name__, cls)
            return AttributeTree.__new__(ncls)

        def __init__(self, eq,  *args, npsi=129,  ** kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__["_eq"] = eq
            self._npsi = npsi

        @cached_property
        def psi_norm(self):
            return np.linspace(0, 1, self._npsi)

        @cached_property
        def psi(self):
            """Poloidal flux {dynamic} [Wb]. """
            psi_scale = self._eq.global_quantities.psi_boundary - self._eq.global_quantities.psi_axis
            return (self.psi_norm)*psi_scale+self._eq.global_quantities.psi_axis

        @cached_property
        def phi(self):
            """Toroidal flux {dynamic} [Wb]."""
            psi_norm = self.psi_norm
            psi_scale = self._eq.global_quantities.psi_boundary-self._eq.global_quantities.psi_axis
            psi = self.psi

            q = Interpolate1D(psi_norm, self.q)
            return [q.integral(0, p) for p in psi_norm]

        @cached_property
        def pressure(self):
            """	Pressure {dynamic} [Pa]"""
            # pprime = self.dpresure_dpsi
            # return [pprime.integral(0, self.psi) for psi in self.psi]
            return NotImplemented

        @cached_property
        def f(self):
            """Diamagnetic function (F=R B_Phi) {dynamic} [T.m]."""
            # ffprime = self.f_df_dpsi
            # f2 = [2*ffprime.integral(0, psi) for psi in self.psi]
            # return np.sqrt(f2)
            return NotImplemented

        @cached_property
        def fpol(self):
            return self.f

        @cached_property
        def fdia(self):
            return self.f

        @cached_property
        def pprime(self):
            """Derivative of pressure w.r.t. psi {dynamic} [Pa.Wb^-1]."""
            return self.dpressure_dpsi

        @cached_property
        def dpressure_dpsi(self):
            """Derivative of pressure w.r.t. psi {dynamic} [Pa.Wb^-1]."""
            return NotImplemented

        @cached_property
        def ffprime(self):
            """	Derivative of F w.r.t. Psi, multiplied with F {dynamic} [T^2.m^2/Wb]. """
            return self.f_df_dpsi

        @cached_property
        def f_df_dpsi(self):
            """	Derivative of F w.r.t. Psi, multiplied with F {dynamic} [T^2.m^2/Wb]. """
            return NotImplemented

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
            return NotImplemented

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
            phi = self.phi
            b0 = self._eq.vacuum_toroidal_field.b0 or 1.0
            return np.sqrt(phi)/np.sqrt(scipy.constants.pi*b0)

        @cached_property
        def rho_tor_norm(self):
            """Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm, is the toroidal flux coordinate at the equilibrium boundary
                (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation) {dynamic} [-]"""
            return self.rho_tor/self.rho_tor[-1]

        @cached_property
        def dpsi_drho_tor(self)	:
            """Derivative of Psi with respect to Rho_Tor {dynamic} [Wb/m]. """
            return self._eq.vacuum_toroidal_field.b0*self.rho/self.q

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
        def dvolume_dpsi(self):
            """Radial derivative of the volume enclosed in the flux surface with respect to Psi {dynamic} [m^3.Wb^-1]. """
            return NotImplemented

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
            return NotImplemented

        @cached_property
        def gm4(self):
            """ Flux surface averaged 1/B^2 {dynamic} [T^-2]	"""
            return NotImplemented

        @cached_property
        @cached_property
        def gm5(self):
            """ Flux surface averaged B^2 {dynamic} [T^2]	"""
            return NotImplemented

        @cached_property
        @cached_property
        def gm6(self):
            """ Flux surface averaged |grad_rho_tor|^2/B^2 {dynamic} [T^-2]	"""
            return NotImplemented

        @cached_property
        def gm7(self):
            """ Flux surface averaged |grad_rho_tor| {dynamic} [-]	"""
            return NotImplemented

        @cached_property
        def gm8(self):
            """ Flux surface averaged R {dynamic} [m]	"""
            return NotImplemented

        @cached_property
        def gm9(self):
            """ Flux surface averaged 1/R {dynamic} [m^-1]          """
            return NotImplemented

    class Profiles2D(AttributeTree):
        @staticmethod
        def __new__(cls,   eq,  *args, **kwargs):
            if cls is not Equilibrium.Profiles2D:
                return super(Equilibrium.Profiles2D, cls).__new__(cls)
            ncls = getattr(eq.__class__, cls.__name__, cls)
            return AttributeTree.__new__(ncls)

        def __init__(self, eq, *args,  ** kwargs):
            super().__init__(*args, **kwargs)
            self._eq = eq

        @cached_property
        def r(self):
            return NotImplemented

        @cached_property
        def z(self):
            return NotImplemented

        def psi(self, *args, **kwargs):
            return NotImplemented

    class Boundary(AttributeTree):
        @staticmethod
        def __new__(cls,   eq,  *args, **kwargs):
            if cls is not Equilibrium.Boundary:
                return super(Equilibrium.Boundary, cls).__new__(cls)
            ncls = getattr(eq.__class__, cls.__name__, cls)
            return AttributeTree.__new__(ncls)

        def __init__(self, eq, *args, ntheta=129,  ** kwargs):
            super().__init__(*args, **kwargs)
            self._eq = eq
            self._ntheta = ntheta

        @cached_property
        def type(self):
            """0 (limiter) or 1 (diverted) {dynamic} """
            return 1

        @cached_property
        def outline(self):
            """ RZ outline of the plasma boundary  """
            boundary = np.array([p for p in self._eq.find_surface(1.0, ntheta=self._ntheta)])

            return AttributeTree({
                "r": boundary[:, 0],
                "z": boundary[:, 1]
            })

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
        @staticmethod
        def __new__(cls,   eq,  *args, **kwargs):
            if cls is not Equilibrium.BoundarySeparatrix:
                return super(Equilibrium.BoundarySeparatrix, cls).__new__(cls)
            ncls = getattr(eq.__class__, cls.__name__, cls)
            return AttributeTree.__new__(ncls)

        def __init__(self, eq, *args,  ** kwargs):
            super().__init__(*args, **kwargs)
            self._eq = eq

    ####################################################################################
    # Plot proflies

    def plot(self, axis=None, *args, levels=40, oxpoints=True, **kwargs):
        """ learn from freegs
        """
        if axis is None:
            axis = plt.gca()

        R = self.profiles_2d.r
        Z = self.profiles_2d.z
        psi = self.profiles_2d.psi

        # if type(levels) is int:
        #     levels = np.linspace(self.global_quantities.psi_axis, self.global_quantities.psi_boundary, levels)

        axis.contour(R, Z, psi, levels=levels, linewidths=0.2)

        lcfs_points = np.array([self.boundary.outline_inner.r,
                                self.boundary.outline_inner.z]).transpose([1, 0])

        axis.add_patch(plt.Polygon(lcfs_points, fill=False, closed=True))

        for idx, p in enumerate(self.boundary.x_point):
            axis.plot(p.r, p.z, 'rx')
            axis.text(p.r, p.z, idx)

        boundary_points = np.array([self.boundary.outline.r, self.boundary.outline.z]).transpose([1, 0])

        axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='dashed',
                                   linewidth=0.5, fill=False, closed=True))
        axis.plot([], [], 'r--', label="Separatrix")

        axis.plot(self.global_quantities.magnetic_axis.r,
                  self.global_quantities.magnetic_axis.z, 'g.', label="Magnetic axis")

        return axis

    def plot_full(self,  profiles=None, profiles_label=None, x_axis="psi_norm", xlabel=r'$(\psi-\psi_{axis})/(\psi_{boundary}-\psi_{axis})$', *args, **kwargs):

        if isinstance(profiles, str):
            profiles = profiles.split(",")
        elif profiles is None:
            profiles = [("q", r"q", r"$[-]$"),
                        ("pprime", r"$p^{\prime}$", r"$[Pa / Wb]$"),
                        ("ffprime", r"$f f^{\prime}$", r"$[T^2.m^2/Wb]$"),
                        ("fpol", r"$f_{pol}$", r"$[T \cdot m]$"),
                        ("pressure", r"pressure", r"$[Pa]$")]

        def fetch_profile(p):
            if isinstance(p, str):
                p = {"key": p, "label": p}
            elif isinstance(p, tuple):
                key, label, unit = p
                p = {"key": key, "label": label, "unit": unit}

            try:
                d = self.profiles_1d[p["key"]]
            except KeyError:
                d = None

            if d is None or d is NotImplemented or len(d) == 0:
                logger.warning(f"Plot profile '{p}' failed!")
            else:
                p["data"] = d

            return p

        profiles = [p for p in map(fetch_profile, profiles) if p.get("data", None) is not None]

        nprofiles = len(profiles)

        if nprofiles == 0:
            return self.plot(*args, **kwargs)

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

        x = self.profiles_1d[x_axis]

        if profiles_label is None:
            profiles_label = profiles

        for idx, p in enumerate(profiles):
            axs[idx, 0].plot(x, p["data"],  label=p["label"])
            unit = p.get("unit", None)
            if unit is not None:
                axs[idx, 0].set_ylabel(unit)
            axs[idx, 0].legend()

        axs[nprofiles-1, 0].set_xlabel(xlabel)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
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
