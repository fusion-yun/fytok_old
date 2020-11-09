import collections
import sys

import freegs
import numpy as np
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles

from ...CoreProfiles import CoreProfiles
from ...Equilibrium import Equilibrium, EqProfiles1D


class EqProfiles1DFreeGS(EqProfiles1D):
    def __init__(self, backend, *args,  ** kwargs):
        super().__init__(*args, **kwargs)
        self._backend = backend

    @property
    def psi_norm(self):
        return self.dimensions

    def pprime(self, psi_norm=None):
        return self._backend.pprime(psi_norm if psi_norm is not None else self.psi_norm)

    def ffprime(self,  psi_norm=None):
        return self._backend.ffprime(psi_norm if psi_norm is not None else self.psi_norm)

    def pressure(self,   psi_norm=None):
        return self._backend.pressure(psi_norm if psi_norm is not None else self.psi_norm)

    def fpol(self,   psi_norm=None):
        return self._backend.fpol(psi_norm if psi_norm is not None else self.psi_norm)

    def q(self, psi_norm=None):
        return self._backend.q(psi_norm if psi_norm is not None else self.psi_norm)

    def f(self,   psi_norm=None):
        return psi_norm if psi_norm is not None else self.psi_norm

    def dvolume_dpsi(self, psi_norm=None):
        return psi_norm if psi_norm is not None else self.psi_norm

    def gm2(self, psi_norm=None):
        return psi_norm if psi_norm is not None else self.psi_norm


class EqProfiles2DFreeGS(Profiles):
    """
        Equilibrium 2D profiles in the poloidal plane.
        @ref: equilibrium.time_slice[itime].profiles_2d
    """

    def __init__(self, backend, dims=None, *args, **kwargs):
        super().__init__(dims or [129, 129], **kwargs)
        self._backend = backend

    def r(self, dims=None):
        return self._backend.R

    def z(self, dims=None):
        return self._backend.Z

    def psi(self, dims=None):
        return self._backend.psi()


class EquilibriumFreeGS(Equilibrium):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, ids=None, *args,  **kwargs):
        super().load(ids, *args, **kwargs)

        self.vacuum_toroidal_field.b0 = self.tokamak.vacuum_toroidal_field.b0
        self.vacuum_toroidal_field.r0 = self.tokamak.vacuum_toroidal_field.r0

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

        self.profiles_1d = EqProfiles1DFreeGS(self._backend)
        self.profiles_2d = EqProfiles2DFreeGS(self._backend)

        return self.entry

    # @property
    # def profiles_1d(self):
    #     if not hasattr(self, "_profiles_1d"):
    #         self._profiles_1d = EqProfiles1DFreeGS(self)
    #     return self._profiles_1d

    @property
    def r(self):
        return self._backend.R

    @property
    def z(self):
        return self._backend.Z

    @property
    def psi(self):
        return self._backend.psi()

    @property
    def psi_norm(self):
        return self._backend.psi_norm

    @property
    def oxpoints(self):
        return freegs.critical.find_critical(self.r, self.z, self.psi)

    def solve(self, core_profiles, constraints=None, fvec=None, **kwargs):
        if fvec is None:
            fvec = 1.0

        if isinstance(core_profiles, CoreProfiles):
            self.vacuum_toroidal_field.b0 = core_profiles.vacuum_toroidal_field.b0
            psi_norm = self.profiles_1d.psi_norm
            profiles = freegs.jtor.ProfilesPprimeFfprime(
                core_profiles.pprime(psi_norm),
                core_profiles.ffprime(psi_norm),
                self.vacuum_toroidal_field.b0 * self.vacuum_toroidal_field.r0)

        elif core_profiles is not None and "Ip" in core_profiles:
            self.vacuum_toroidal_field.b0 = fvec / self.vacuum_toroidal_field.r0
            profiles = freegs.jtor.jtor.ConstrainPaxisIp(
                core_profiles["pressure"],  # Plasma pressure on axis [Pascals]
                core_profiles["Ip"],        # Plasma current [Amps]
                fvec)
        else:
            logger.warning(f"Using default profile pressure=1e3 Ip=1e6")
            self.vacuum_toroidal_field.b0 = fvec / self.vacuum_toroidal_field.r0
            profiles = freegs.jtor.ConstrainPaxisIp(
                1e3,  # Plasma pressure on axis [Pascals]
                1e6,  # Plasma current [Amps]
                fvec)

        constrain = freegs.control.constrain(** (constraints or {}))

        logger.debug(f"Solve Equilibrium [{self.__class__.__name__}] Start")

        freegs.solve(self._backend, profiles, constrain)

        logger.debug(f"Solve Equilibrium [{self.__class__.__name__}] End")

        self.update_global_quantities()
        return

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
        self.global_quantities.psi_axis = self._backend.psi_axis
        # Poloidal flux at the selected plasma boundary {dynamic} [Wb]. 
        self.global_quantities.psi_boundary = self._backend.psi_bndry
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
