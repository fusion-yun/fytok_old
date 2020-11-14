import collections
import sys
from functools import cached_property
import freegs
from freegs.critical import find_separatrix
import numpy as np
from spdm.util.Interpolate import derivate, integral, interpolate, Interpolate2D
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles
from spdm.util.AttributeTree import _next_
from ...CoreProfiles import CoreProfiles
from ...Equilibrium import Equilibrium


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

        try:
            freegs.solve(self._backend, profiles, constraints)
        except ValueError as error:
            raise RuntimeError(f"Solve G-S equation failed [{self.__class__.__name__}]! {error}")

    class Profiles1D(Equilibrium.Profiles1D):
        def __init__(self, eq, *args, **kwargs):
            super().__init__(eq, *args, **kwargs)
            self.__dict__['backend'] = eq._backend

        @cached_property
        def pprime(self):
            return self._backend.pprime(self.psi_norm)

        @cached_property
        def pressure(self):
            return self._backend.pressure(self.psi_norm)

        @cached_property
        def f(self):
            return self._backend.fpol(self.psi_norm)

        @cached_property
        def f_df_dpsi(self):
            return self._backend.ffprime(self.psi_norm)

        @cached_property
        def q(self):
            return self._backend.q(self.psi_norm)

    class Profiles2D(Equilibrium.Profiles2D):
        def __init__(self, eq, *args, **kwargs):
            super().__init__(eq, *args, **kwargs)
            self.__dict__['_backend'] = eq._backend

        @cached_property
        def r(self):
            return self._backend.R

      
        @cached_property
        def z(self):
            return self._backend.Z

      
        @cached_property
        def psi(self):
            return self._backend.psiRZ(self.r,self.z)

        @cached_property
        def theta(self):
            return NotImplemented

        @cached_property
        def phi(self):
            return NotImplemented

        @cached_property
        def b_field_r(self):
            return self._backend.Br(self.r,self.z)

        @cached_property
        def b_field_z(self):
            return self._backend.BZ(self.r,self.z)

        @cached_property
        def b_field_tor(self):
            return self._backend.Btor(self.r,self.z)

    class GlobalQuantities(Equilibrium.GlobalQuantities):
        def __init__(self, eq, *args, **kwargs):
            super().__init__(eq, *args, **kwargs)
            self.__dict__['_backend'] = eq._backend

        @property
        def beta_pol(self):
            # Poloidal beta. Defined as betap = 4 int(p dV) / [R_0 * mu_0 * Ip^2] {dynamic} [-]
            return self._eq._backend.poloidalBeta()

        @property
        def beta_tor(self):
            # Toroidal beta, defined as the volume-averaged total perpendicular pressure divided by (B0^2/(2*mu0)), i.e. beta_toroidal = 2 mu0 int(p dV) / V / B0^2 {dynamic} [-]
            return NotImplemented

        @property
        def beta_normal(self):
            # Normalised toroidal beta, defined as 100 * beta_tor * a[m] * B0 [T] / ip [MA] {dynamic} [-]
            return NotImplemented

        @property
        def ip(self):
            # Plasma current (toroidal component). Positive sign means anti-clockwise when viewed from above. {dynamic} [A].
            return self._eq._backend.plasmaCurrent()

        @property
        def li_3(self):
            # Internal inductance {dynamic} [-]
            return NotImplemented

        @property
        def volume(self):
            # Total plasma volume {dynamic} [m^3]
            return self._eq._backend.plasmaVolume()

        @property
        def area(self):
            # Area of the LCFS poloidal cross section {dynamic} [m^2]
            return NotImplemented

        @property
        def surface(self):
            # Surface area of the toroidal flux surface {dynamic} [m^2]
            return NotImplemented

        @property
        def length_pol(self):
            # Poloidal length of the magnetic surface {dynamic} [m]
            return NotImplemented

        @property
        def psi_axis(self):
            # Poloidal flux at the magnetic axis {dynamic} [Wb].
            return self._backend.psi_axis

        @property
        def psi_boundary(self):
            # Poloidal flux at the selected plasma boundary {dynamic} [Wb].
            return self._backend.psi_bndry

        # @property
        # def magnetic_axis(self):
        #     # Magnetic axis position and toroidal field	structure
        #     return AttributeTree({"r":  opt[0][0],
        #                           "z":  opt[0][1],
        #                           "b_field_tor":  self._backend.Btor(opt[0][0], opt[0][1])
        #                           })

        @property
        def q_axis(self):
            # q at the magnetic axis {dynamic} [-].
            return NotImplemented

        @property
        def q_95(self):
            # q at the 95% poloidal flux surface (IMAS uses COCOS=11: only positive when toroidal current and magnetic field are in same direction) {dynamic} [-].
            return NotImplemented

        @property
        def q_min(self):
            # Minimum q value and position	structure
            return NotImplemented

        @property
        def energy_mhd(self):
            # Plasma energy content(self): return 3/2 * int(p,dV) with p being the total pressure (thermal + fast particles) [J]. Time-dependent; Scalar {dynamic} [J]
            return NotImplemented

    class Boundary(Equilibrium.Boundary):
        def __init__(self, eq, *args, opt=None, xpt=None, ntheta=129, **kwargs):
            super().__init__(eq, *args, **kwargs)
            self.__dict__['_backend'] = eq._backend

            R = self._backend.R
            Z = self._backend.Z
            psi = self._backend.psiRZ(R, Z)

            # find magnetic axis (o-point) and x-points
            if opt is None or xpt is None:
                opt, xpt = freegs.critical.find_critical(R, Z, psi)

            if not opt or not xpt:
                raise RuntimeError(f"Can not find O-point or X-points!")

            r, z, v = opt[0]

            self._eq.global_quantities.magnetic_axis.r = opt[0][0]
            self._eq.global_quantities.magnetic_axis.z = opt[0][1]
            self._eq.global_quantities.magnetic_axis.b_field_tor = self._backend.Btor(opt[0][0], opt[0][1])

            for r, z, _ in xpt:
                self.x_point[_next_] = {"r": r, "z": z}

            # Value of the poloidal flux at which the boundary is taken {dynamic} [Wb]
            self.psi = self._backend.psi_bndry
            # Value of the normalised poloidal flux at which the boundary is taken (typically 99.x %),
            # the flux being normalised to its value at the separatrix {dynamic}
            self.psi_norm = 0.99

            lcfs = np.array(freegs.critical.find_separatrix(
                self._backend, opoint=opt, xpoint=xpt, psi=psi, ntheta=ntheta))
            # lcfs = self._eq._backend.separatrix(ntheta=ntheta)
            self.outline.r = lcfs[:, 0]
            self.outline.z = lcfs[:, 1]

            self.geometric_axis.r = (min(lcfs[:, 0])+max(lcfs[:, 0]))/2
            self.geometric_axis.z = (min(lcfs[:, 1])+max(lcfs[:, 1]))/2
