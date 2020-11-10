
import collections
import functools

import matplotlib.pyplot as plt
import numpy as np
import scipy
from spdm.data.Entry import open_entry
from spdm.util.AttributeTree import AttributeTree
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles1D, Profiles2D
from spdm.util.sp_export import sp_find_module

from .PFActive import PFActive
from .Wall import Wall
#  psi phi pressure f dpressure_dpsi f_df_dpsi j_parallel q magnetic_shear r_inboard r_outboard rho_tor rho_tor_norm dpsi_drho_tor geometric_axis elongation triangularity_upper triangularity_lower volume rho_volume_norm dvolume_dpsi dvolume_drho_tor area darea_dpsi surface trapped_fraction gm1 gm2 gm3 gm4 gm5 gm6 gm7 gm8 gm9 b_field_max beta_pol mass_density


class EqProfiles1D(Profiles1D):
    def __init__(self,  *args, dimensions=None,  ** kwargs):
        npoints = 129
        super().__init__(dimensions or np.linspace(1.0/(npoints+1), 1.0, npoints), *args, **kwargs)

    @property
    def psi_norm(self):
        return self.dimensions

    def psi(self,   psi_norm=None):
        """Poloidal flux {dynamic} [Wb]. This quantity is COCOS-dependent, with the following transformation : """
        return NotImplemented

    def phi(self,   psi_norm=None):
        """Toroidal flux {dynamic} [Wb]. This quantity is COCOS-dependent, with the following transformation :"""
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
        return NotImplemented

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
        return NotImplemented

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


class EqProfiles2D(Profiles2D):
    def __init__(self,  *args,  ** kwargs):
        super().__init__(*args, **kwargs)


class Equilibrium(AttributeTree):
    """
        Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.
        imas dd version 3.28
        ids=equilibrium
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
        self.coordinate_system.grid.dim1 = np.linspace(min(lim_r), max(lim_r), nr)
        self.coordinate_system.grid.dim2 = np.linspace(min(lim_z), max(lim_z), nz)

    @property
    def oxpoints(self):
        return NotImplemented

    @property
    def r(self):
        return NotImplemented

    @property
    def z(self):
        return NotImplemented

    @property
    def psi(self):
        return NotImplemented

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

        # self.update_global_quantities()

    def update_global_quantities(self):

        # Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2] {dynamic} [-]	FLT_0D
        self.global_quantities.beta_pol = NotImplemented
        # Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2 {dynamic} [-]	FLT_0D
        self.global_quantities.beta_tor = NotImplemented
        # Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA] {dynamic} [-]	FLT_0D
        self.global_quantities.beta_normal = NotImplemented
        # Plasma current (toroidal component). Positive sign means anti-clockwise when viewed from above. {dynamic} [A]. This quantity is COCOS-dependent, with the following transformation :
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
        # Poloidal flux at the magnetic axis {dynamic} [Wb]. This quantity is COCOS-dependent, with the following transformation :
        self.global_quantities.psi_axis = NotImplemented
        # Poloidal flux at the selected plasma boundary {dynamic} [Wb]. This quantity is COCOS-dependent, with the following transformation :
        self.global_quantities.psi_boundary = NotImplemented
        # Magnetic axis position and toroidal field	structure
        self.global_quantities.magnetic_axis = NotImplemented
        # q at the magnetic axis {dynamic} [-]. This quantity is COCOS-dependent, with the following transformation :
        self.global_quantities.q_axis = NotImplemented
        # q at the 95% poloidal flux surface (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction) {dynamic} [-]. This quantity is COCOS-dependent, with the following transformation :
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

        R = self.r
        Z = self.z
        Psi = self.psi

        levels = np.linspace(np.amin(Psi), np.amax(Psi), levels)

        axis.contour(R, Z, Psi, levels=levels, linewidths=0.2)

        if oxpoints:
            opts, xpts = self.oxpoints

            if len(opts) > 0:
                axis.plot([p[0] for p in opts], [p[1] for p in opts], 'g.', label="O-points")

            if len(xpts) > 0:
                axis.plot([p[0] for p in xpts], [p[1] for p in xpts], 'rx', label="X-points")
                psi_bndry = xpts[0][2]
                axis.contour(R, Z, Psi, levels=[psi_bndry], colors='r', linestyles='dashed', linewidths=0.4)
                axis.plot([], [], 'r--', label="Separatrix")

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
