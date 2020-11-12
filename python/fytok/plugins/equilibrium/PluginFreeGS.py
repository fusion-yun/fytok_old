import collections
import sys

import freegs
import numpy as np
from spdm.util.Interpolate import derivate, integral, interpolate
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles
from spdm.util.AttributeTree import _next_
from ...CoreProfiles import CoreProfiles
from ...Equilibrium import EqProfiles1D, EqProfiles2D, Equilibrium


class EqProfiles1DFreeGS(EqProfiles1D):
    def __init__(self, backend, *args,  ** kwargs):
        super().__init__(*args, **kwargs)
        self._backend = backend

    def psi_norm(self, psi_norm):
        return self._backend.psi_norm(psi_norm)

    def pprime(self, psi_norm):
        return self._backend.pprime(psi_norm)

    def ffprime(self,  psi_norm):
        return self._backend.ffprime(psi_norm)

    def pressure(self,   psi_norm):
        return self._backend.pressure(psi_norm)

    def fpol(self,   psi_norm):
        return self._backend.fpol(psi_norm)

    def q(self, psi_norm):
        return self._backend.q(psi_norm)

    def f(self, psi_norm):
        return self._backend.fpol(psi_norm)

    def rho_tor(self,  psi_norm):
        """Toroidal flux coordinate. The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0 {dynamic} [m]"""
        logger.debug(f"FIXME: NOT IMPLEMENTED!")
        return interpolate(self.x, self.x)(psi_norm)

    def dpsi_drho_tor(self,  psi_norm)	:
        """Derivative of Psi with respect to Rho_Tor {dynamic} [Wb/m]. """
        logger.debug(f"FIXME: NOT IMPLEMENTED!")
        return interpolate(self.x, self.x)(psi_norm)

    def volume(self,  psi_norm):
        """Volume enclosed in the flux surface {dynamic} [m^3]"""
        return NotImplemented

    def dvolume_dpsi(self,  psi_norm):
        """Radial derivative of the volume enclosed in the flux surface with respect to Psi {dynamic} [m^3.Wb^-1]. """
        logger.debug(f"FIXME: NOT IMPLEMENTED!")
        return np.zeros(shape=psi_norm.shape)

    def dvolume_drho_tor(self,  psi_norm)	:
        """Radial derivative of the volume enclosed in the flux surface with respect to Rho_Tor {dynamic} [m^2]"""
        return self.dvolume_dpsi(psi_norm) * self.dpsi_drho_tor(psi_norm)


class EqProfiles2DFreeGS(EqProfiles2D):
    """
        Equilibrium 2D profiles in the poloidal plane.
        @ref: equilibrium.time_slice[itime].profiles_2d
    """

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def R(self):
        return self._eq._backend.R

    @property
    def Z(self):
        return self._eq._backend.Z

    def psi(self, R=None, Z=None):
        return self._eq._backend.psiRZ(R if R is not None else self.R, Z if Z is not None else self.Z)

    def b_field_r(self,  R=None, Z=None):
        return self._eq._backend.Br(R if R is not None else self.R, Z if Z is not None else self.Z)

    def b_field_z(self,  R=None, Z=None):
        return self._eq._backend.Bz(R if R is not None else self.R, Z if Z is not None else self.Z)

    def b_field_tor(self,  R=None, Z=None):
        return self._eq._backend.Btor(R if R is not None else self.R, Z if Z is not None else self.Z)


class EquilibriumFreeGS(Equilibrium):
    def __init__(self,  config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        eq_wall = freegs.machine.Wall(self.tokamak.wall.limiter.outline.r,
                                      self.tokamak.wall.limiter.outline.z)

        eq_coils = []

        for coil in self.tokamak.pf_active.coil:
            t_coil = freegs.machine.Coil(
                coil.r+coil.width/2,
                coil.z+coil.height/2,
                turns=coil.turns)
            eq_coils.append((coil.name, t_coil))

        tokamak = freegs.machine.Machine(eq_coils, wall=eq_wall)

        dim1 = self.coordinate_system.grid.dim1
        dim2 = self.coordinate_system.grid.dim2

        self._backend = freegs.Equilibrium(
            tokamak=tokamak,
            Rmin=min(dim1), Rmax=max(dim1),
            Zmin=min(dim2), Zmax=max(dim2),
            nx=len(dim1), ny=len(dim2),
            boundary=freegs.boundary.freeBoundaryHagenow)

        self.profiles_1d = EqProfiles1DFreeGS(self)
        self.profiles_2d = EqProfiles2DFreeGS(self)

    # @property
    # def profiles_1d(self):
    #     if not hasattr(self, "_profiles_1d"):
    #         self._profiles_1d = EqProfiles1DFreeGS(self)
    #     return self._profiles_1d

    @property
    def psi_axis(self):
        return self._backend.psi_axis

    @property
    def psi_boundary(self):
        return self._backend.psi_bndry

    @property
    def oxpoints(self):
        return freegs.critical.find_critical(self.r, self.z, self.psi)

    def solve(self, core_profiles=None, constraints=None,  **kwargs):
        if not isinstance(core_profiles, CoreProfiles):
            core_profiles = CoreProfiles(core_profiles)

        if "pprime" in core_profiles.profiles_1d and "ffprime" in core_profiles.profiles_1d:
            profiles = freegs.jtor.ProfilesPprimeFfprime(
                core_profiles.profiles_1d.pprime,
                core_profiles.profiles_1d.ffprime,
                core_profiles.vacuum_toroidal_field.b0 * core_profiles.vacuum_toroidal_field.r0)
        elif "ip" in core_profiles.global_quantities and "pressure" in core_profiles.global_quantities:
            profiles = freegs.jtor.jtor.ConstrainPaxisIp(
                core_profiles.global_quantities.pressure,                  # Plasma pressure on axis [Pascals]
                core_profiles.global_quantities.ip,         # Plasma current [Amps]
                core_profiles.vacuum_toroidal_field.b0 * core_profiles.vacuum_toroidal_field.r0)
        else:
            logger.debug(f"Using default profile pressure=1e3 Ip=1e6")
            profiles = freegs.jtor.ConstrainPaxisIp(
                1e3,  # Plasma pressure on axis [Pascals]
                1e6,  # Plasma current [Amps]
                kwargs.get("fvec", 1.0))

        constraints = freegs.control.constrain(** (constraints or {}))

        freegs.solve(self._backend, profiles, constraints)

    def update_global_quantities(self):
        # Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2] {dynamic} [-]
        self.global_quantities.beta_pol = self._backend.poloidalBeta()
        # Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2 {dynamic} [-]
        self.global_quantities.beta_tor = NotImplemented
        # Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA] {dynamic} [-]
        self.global_quantities.beta_normal = NotImplemented
        # Plasma current (toroidal component). Positive sign means anti-clockwise when viewed from above. {dynamic} [A].
        self.global_quantities.ip = self._backend.plasmaCurrent()
        # Internal inductance {dynamic} [-]
        self.global_quantities.li_3 = NotImplemented
        # Total plasma volume {dynamic} [m^3]
        self.global_quantities.volume = self._backend.plasmaVolume()
        # Area of the LCFS poloidal cross section {dynamic} [m^2]
        self.global_quantities.area = NotImplemented
        # Surface area of the toroidal flux surface {dynamic} [m^2]
        self.global_quantities.surface = NotImplemented
        # Poloidal length of the magnetic surface {dynamic} [m]
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

    def update_surface(self, nbdry=128):
        """
            base on  freegs.critical
        """
        # Value of the poloidal flux at which the boundary is taken {dynamic} [Wb]
        self.boundary.psi = self._backend.psi_bndry
        # Value of the normalised poloidal flux at which the boundary is taken (typically 99.x %),
        # the flux being normalised to its value at the separatrix {dynamic}
        self.boundary.psi_norm = 0.99

        psi = self.profiles_2d.psi()

        # find magnetic axis (o-point) and x-points
        opt, xpt = freegs.critical.find_critical(R, Z, psi)
        if opt:
            r, z, v = opt[0]
            self.global_quantities.magnetic_axis.r = r
            self.global_quantities.magnetic_axis.z = z
            self.global_quantities.psi_axis = v

        if xpt:
            for r, z, _ in xpt:
                self.boundary.xpoint[_next_] = {"r": r, "z": z}

        isoflux = freegs.find_separatrix(self._backend, opoint=opt, xpoint=xpt, psi=psi, ntheta=nbdry)

        lcfs = self._backend.separatrix(ntheta=nbdry)
        self.boundary.outline.r = lcfs[:, 0]
        self.boundary.outline.z = lcfs[:, 1]

        self.boundary.geometric_axis.r = (min(lcfs[:, 0])+max(lcfs[:, 0]))/2
        self.boundary.geometric_axis.z = (min(lcfs[:, 1])+max(lcfs[:, 1]))/2
