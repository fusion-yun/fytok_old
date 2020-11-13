
import collections
import functools

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import arctan2, cos, sin, sqrt
from spdm.data.Entry import open_entry
from spdm.util.AttributeTree import AttributeTree, _next_
from spdm.util.Interpolate import (Interpolate2D, derivate, find_root, find_critical,
                                   integral, interpolate)
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles
from spdm.util.sp_export import sp_find_module
from sympy import Point, Polygon
from scipy.optimize import root_scalar

from .PFActive import PFActive
from .Wall import Wall

#  psi phi pressure f dpressure_dpsi f_df_dpsi j_parallel q magnetic_shear r_inboard r_outboard rho_tor rho_tor_norm dpsi_drho_tor geometric_axis elongation triangularity_upper triangularity_lower volume rho_volume_norm dvolume_dpsi dvolume_drho_tor area darea_dpsi surface trapped_fraction gm1 gm2 gm3 gm4 gm5 gm6 gm7 gm8 gm9 b_field_max beta_pol mass_density


class EqProfiles1D(Profiles):
    def __init__(self, dims=129,  *args, psi_boundary=1.0,  ** kwargs):
        super().__init__(dims, *args, **kwargs)
        self.__dict__["_psi_boundary"] = psi_boundary

    @property
    def psi_norm(self):
        return self.x

    def psi(self,   psi_norm=None):
        """Poloidal flux {dynamic} [Wb]. """
        return psi_norm * self._psi_boundary if psi_norm is not None else self.x*self._psi_boundary

    def phi(self,   psi_norm=None):
        """Toroidal flux {dynamic} [Wb]."""
        return NotImplemented

    def pressure(self,   psi_norm=None):
        """	Pressure {dynamic} [Pa]"""
        return NotImplemented

    def f(self,   psi_norm=None):
        """Diamagnetic function (F=R B_Phi) {dynamic} [T.m]."""
        return NotImplemented

    def pprime(self, psi_norm=None):
        """Derivative of pressure w.r.t. psi {dynamic} [Pa.Wb^-1]."""
        return NotImplemented

    def dpressure_dpsi(self,   psi_norm=None):
        """Derivative of pressure w.r.t. psi {dynamic} [Pa.Wb^-1]."""
        return self.pprime(psi_norm)

    def ffprime(self,  psi_norm=None):
        """	Derivative of F w.r.t. Psi, multiplied with F {dynamic} [T^2.m^2/Wb]. """
        return NotImplemented

    def f_df_dpsi(self,  psi_norm=None):
        """	Derivative of F w.r.t. Psi, multiplied with F {dynamic} [T^2.m^2/Wb]. """
        return self.ffprime(psi_norm)

    def j_tor(self,  psi_norm=None):
        """Flux surface averaged toroidal current density = average(j_tor/R) / average(1/R) {dynamic} [A.m^-2]."""
        return NotImplemented

    def j_parallel(self,  psi_norm=None):
        """Flux surface averaged parallel current density = average(j.B) / B0, where B0 = Equilibrium/Global/Toroidal_Field/B0 {dynamic} [A/m^2].  """
        return NotImplemented

    def q(self,  psi_norm=None):
        """Safety factor (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction) {dynamic} [-]. """
        return NotImplemented

    def magnetic_shear(self,  psi_norm=None):
        """Magnetic shear, defined as rho_tor/q . dq/drho_tor {dynamic} [-]	 """
        return NotImplemented

    def r_inboard(self,  psi_norm=None):
        """Radial coordinate (major radius) on the inboard side of the magnetic axis {dynamic} [m]"""
        return NotImplemented

    def r_outboard(self,  psi_norm=None):
        """Radial coordinate (major radius) on the outboard side of the magnetic axis {dynamic} [m]"""
        return NotImplemented

    def rho_tor(self,  psi_norm=None):
        """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0 {dynamic} [m]"""
        return NotImplemented

    def rho_tor_norm(self,  psi_norm=None):
        """Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm, is the toroidal flux coordinate at the equilibrium boundary
            (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation) {dynamic} [-]"""
        rho_tor = self.rho_tor(psi_norm)
        return rho_tor/rho_tor[-1]

    def dpsi_drho_tor(self,  psi_norm=None)	:
        """Derivative of Psi with respect to Rho_Tor {dynamic} [Wb/m]. """
        return NotImplemented

    def volume(self,  psi_norm=None):
        """Volume enclosed in the flux surface {dynamic} [m^3]"""
        return NotImplemented

    def rho_volume_norm(self,  psi_norm=None)	:
        """Normalised square root of enclosed volume (radial coordinate). The normalizing value is the enclosed volume at the equilibrium boundary
             (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation) {dynamic} [-]"""
        return NotImplemented

    def dvolume_dpsi(self,  psi_norm=None):
        """Radial derivative of the volume enclosed in the flux surface with respect to Psi {dynamic} [m^3.Wb^-1]. """
        return NotImplemented

    def dvolume_drho_tor(self,  psi_norm=None)	:
        """Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor {dynamic} [m^2]"""
        return self.dvolume_dpsi(psi_norm)*self.dpsi_drho_tor(psi_norm)

    def area(self,  psi_norm=None):
        """Cross-sectional area of the flux surface {dynamic} [m^2]"""
        return NotImplemented

    def darea_dpsi(self,  psi_norm=None):
        """Radial derivative of the cross-sectional area of the flux surface with respect to psi {dynamic} [m^2.Wb^-1]. """
        return NotImplemented

    def darea_drho_tor(self,  psi_norm=None)	:
        """Radial derivative of the cross-sectional area of the flux surface with respect to rho_tor {dynamic} [m]"""
        return NotImplemented

    def surface(self,  psi_norm=None):
        """Surface area of the toroidal flux surface {dynamic} [m^2]"""
        return NotImplemented

    def trapped_fraction(self,  psi_norm=None)	:
        """Trapped particle fraction {dynamic} [-]"""
        return NotImplemented

    def b_field_max(self,  psi_norm=None):
        """Maximum(modulus(B)) on the flux surface (always positive, irrespective of the sign convention for the B-field direction) {dynamic} [T]"""
        return NotImplemented

    def beta_pol(self,  psi_norm=None):
        """Poloidal beta profile. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2] {dynamic} [-]"""
        return NotImplemented

    def mass_density(self,  psi_norm=None):
        """Mass density {dynamic} [kg.m^-3]"""
        return NotImplemented

    def fpol(self,   psi_norm=None):
        return NotImplemented

    def gm1(self, psi_norm=None):
        """ Flux surface averaged 1/R^2 {dynamic} [m^-2]  """
        return NotImplemented

    def gm2(self, psi_norm=None):
        """ Flux surface averaged |grad_rho_tor|^2/R^2 {dynamic} [m^-2] """
        return NotImplemented

    def gm3(self, psi_norm=None):
        """ Flux surface averaged |grad_rho_tor|^2 {dynamic} [-]	"""
        return NotImplemented

    def gm4(self, psi_norm=None):
        """ Flux surface averaged 1/B^2 {dynamic} [T^-2]	"""
        return NotImplemented

    def gm5(self, psi_norm=None):
        """ Flux surface averaged B^2 {dynamic} [T^2]	"""
        return NotImplemented

    def gm6(self, psi_norm=None):
        """ Flux surface averaged |grad_rho_tor|^2/B^2 {dynamic} [T^-2]	"""
        return NotImplemented

    def gm7(self, psi_norm=None):
        """ Flux surface averaged |grad_rho_tor| {dynamic} [-]	"""
        return NotImplemented

    def gm8(self, psi_norm=None):
        """ Flux surface averaged R {dynamic} [m]	"""
        return NotImplemented

    def gm9(self, psi_norm=None):
        """ Flux surface averaged 1/R {dynamic} [m^-1]          """
        return NotImplemented


class EqProfiles2D(Profiles):
    def __init__(self, eq, *args,  ** kwargs):
        super().__init__(*args, **kwargs)
        self._eq = eq

    @property
    def R(self):
        return NotImplemented

    @property
    def Z(self):
        return NotImplemented

    def psi(self, *args, **kwargs):
        return NotImplemented

    def psi_norm(self, *args, **kwargs):
        return (self.psi(*args, **kwargs)-self._eq.global_quantities.psi_axis)\
            / (self._eq.global_quantities.psi_boundary-self._eq.global_quantities.psi_axis)


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
        # self.global_quantities

    @property
    def fvec(self):
        return self.tokamak.vacuum_toroidal_field.r0() * self.tokamak.vacuum_toroidal_field.b0()

    def solve(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, *args, time=0.0, **kwargs):
        self.time = time

        logger.debug(f"Solve Equilibrium [{self.__class__.__name__}] at time={self.time} : Start")

        self.solve(*args, **kwargs)

        logger.debug(f"Solve Equilibrium [{self.__class__.__name__}] at time={self.time} : End")

        self.update_global_quantities()

    def update_global_quantities(self):

        # Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2] {dynamic} [-]	FLT_0D
        self.global_quantities.beta_pol = NotImplemented
        # Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2 {dynamic} [-]	FLT_0D
        self.global_quantities.beta_tor = NotImplemented
        # Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA] {dynamic} [-]	FLT_0D
        self.global_quantities.beta_normal = NotImplemented
        # Plasma current (toroidal component). Positive sign means anti-clockwise when viewed from above. {dynamic} [A].
        self.global_quantities.ip = NotImplemented
        # Internal inductance {dynamic} [-]	FLT_0D
        self.global_quantities.li_3 = NotImplemented
        # Total plasma volume {dynamic} [m^3]	FLT_0D
        self.global_quantities.volume = NotImplemented
        # Area of the LCFS poloidal cross section {dynamic} [m^2]	FLT_0D
        self.global_quantities.area = NotImplemented
        # Surface area of the toroidal flux surface {dynamic} [m^2]	FLT_0D
        self.global_quantities.surface = NotImplemented
        # Poloidal length of the magnetic surface {dynamic} [m]	FLT_0D
        self.global_quantities.length_pol = NotImplemented
        # Poloidal flux at the magnetic axis {dynamic} [Wb].
        self.global_quantities.psi_axis = NotImplemented
        # Poloidal flux at the selected plasma boundary {dynamic} [Wb].
        self.global_quantities.psi_boundary = NotImplemented
        # Magnetic axis position and toroidal field	structure
        self.global_quantities.magnetic_axis = NotImplemented
        # q at the magnetic axis {dynamic} [-].
        self.global_quantities.q_axis = NotImplemented
        # q at the 95% poloidal flux surface (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction) {dynamic} [-].
        self.global_quantities.q_95 = NotImplemented
        # Minimum q value and position	structure
        self.global_quantities.q_min = NotImplemented
        # Plasma energy content = 3/2 * int(p,dV) with p being the total pressure (thermal + fast particles) [J]. Time-dependent; Scalar {dynamic} [J]
        self.global_quantities.energy_mhd = NotImplemented

    def plot(self, axis=None, *args, levels=40, oxpoints=True, **kwargs):
        """ learn from freegs
        """
        if axis is None:
            axis = plt.gca()

        R = self.profiles_2d.R
        Z = self.profiles_2d.Z
        psi = self.profiles_2d.psi_norm()

        # if type(levels) is int:
        #     levels = np.linspace(self.global_quantities.psi_axis, self.global_quantities.psi_boundary, levels)

        axis.contour(R, Z, psi, levels=levels, linewidths=0.2)

        lcfs_points = np.array([self.boundary.outline_inner.r,
                                self.boundary.outline_inner.z]).transpose([1, 0])

        axis.add_patch(plt.Polygon(lcfs_points, fill=False, closed=True))

        axis.plot(self.global_quantities.magnetic_axis.r,
                  self.global_quantities.magnetic_axis.z, 'g.', label="Magnetic axis")

        for idx, p in enumerate(self.boundary.x_point):
            axis.plot(p.r, p.z, 'rx', label="X-points")
            axis.text(p.r, p.z, idx)

        boundary_points = np.array([self.boundary.outline.r, self.boundary.outline.z]).transpose([1, 0])

        axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='dashed',
                                   linewidth=0.5, fill=False, closed=True))
        axis.plot([], [], 'r--', label="Separatrix")
        axis.axis("scaled")
        return axis

    def plot_full(self,  profiles=None, profiles_label=None, x_axis="psi_norm", xlabel=r'$\psi_{norm}$', *args, **kwargs):

        if isinstance(profiles, str):
            profiles.split(" ")
        elif profiles is None:
            profiles = ["q", "pprime", "ffprime", "fpol", "pressure"]
            profiles_label = [r"q", r"$p^{\prime}$",  r"$f f^{\prime}$", r"$f_{pol}$", r"pressure"]

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

        for idx, pname in enumerate(profiles):
            y = self.profiles_1d[pname](x)
            axs[idx, 0].plot(x, y,  label=profiles_label[idx])
            # axs[idx, 0].set_ylabel(profiles_label[idx])
            axs[idx, 0].legend()

        axs[nprofiles-1, 0].set_xlabel(xlabel)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        return fig

    # def update_surface(self, psi_norm=None):
    #     """
    #         learn from freegs.critical
    #     """
    #     self._R = self.profiles_2d.R
    #     self._Z = self.profiles_2d.Z
    #     self._psi = self.profiles_2d.psi(R, Z)

    #     self.upate_xpoints()

    def update_boundary(self, R=None, Z=None, psi=None, psi_norm=0.99, nbrdy=128):
        if R is None:
            R = self.profiles_2d.R
        if Z is None:
            Z = self.profiles_2d.Z
        if psi is None:
            psi = self.profiles_2d.psi(R, Z)

        if isinstance(psi, Interpolate2D):
            psi = Interpolate2D(R, Z, psi)

        opoints, xpoints = self.find_oxpoints(R, Z, psi)

        self.global_quantities.magnetic_axis.r = opoints[0][0]

        self.global_quantities.magnetic_axis.z = opoints[0][1]

        for r, z, _ in xpoints:
            self.boundary.x_point[_next_] = {"r": r, "z": z}

        self.global_quantities.psi_axis = opoints[0][2]
        self.global_quantities.psi_boundary = xpoints[0][2]

        boundary = np.array([p for p in self.find_surface(1.0, npoints=nbrdy)])

        self.boundary.outline.r = boundary[:, 0]
        self.boundary.outline.z = boundary[:, 1]
        return

    def find_surface(self, psival, psi_norm=None,  r0=None, z0=None, r1=None, z1=None, npoints=128):
        if psi_norm is None:
            psi_norm = Interpolate2D(self.profiles_2d.R[:, 0], self.profiles_2d.Z[0, :], self.profiles_2d.psi_norm())

        if r0 == None:
            r0 = self.global_quantities.magnetic_axis.r
            z0 = self.global_quantities.magnetic_axis.z
            r1 = self.boundary.x_point[0]["r"]
            z1 = self.boundary.x_point[0]["z"]

        # logger.debug(psival)
        theta0 = arctan2(r1 - r0, z1 - z0)
        R = sqrt((r1-r0)**2+(z1-z0)**2)

        def f(r, x0, x1, val):
            return psi_norm((1.0-r)*x0[0]+r*x1[0], (1.0-r)*x0[1]+r*x1[1]) - val

        for theta in np.linspace(0, scipy.constants.pi*2.0, npoints)+theta0:
            r1 = r0 + R * sin(theta)
            z1 = z0 + R * cos(theta)
            try:
                sol = root_scalar(f, bracket=[0, 1], args=([r0, z0], [r1, z1], psival), method='brentq')
            except ValueError:
                continue
            if not sol.converged:
                continue
            r = sol.root
            yield (1.0-r)*r0+r*r1, (1.0-r)*z0+r*z1

    def find_oxpoints(self, R, Z, psi):
        if not isinstance(psi, Interpolate2D):
            psi = Interpolate2D(R[:, 0], Z[0, :], psi)

        limiter_points = np.array([self.tokamak.wall.limiter.outline.r,
                                   self.tokamak.wall.limiter.outline.z]).transpose([1, 0])
        limiter_polygon = Polygon(*map(Point, limiter_points))

        opoint = []
        xpoint = []

        for r, z, tag in find_critical(psi):
            if not limiter_polygon.encloses(Point(r, z)):
                continue

            if tag < 0.0:  # saddle/X-point
                xpoint.append((r, z, psi(r, z)))
            else:  # extrema/ O-point
                opoint.append((r, z, psi(r, z)))

        Rmid = 0.5*(R[-1, 0] + R[0, 0])
        Zmid = 0.5*(Z[0, -1] + Z[0, 0])
        opoint.sort(key=lambda x: (x[0] - Rmid)**2 + (x[1] - Zmid)**2)
        psi_axis = opoint[0][2]
        xpoint.sort(key=lambda x: (x[2] - psi_axis)**2)

        return opoint, xpoint
