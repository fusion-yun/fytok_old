import collections
import sys
import pprint
import freegs
import numpy as np
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles1D, Profiles2D
from spdm.util.LazyProxy import LazyProxy

from ...Equilibrium import Equilibrium, EqProfiles1D
from ...CoreProfiles import CoreProfiles


class EqProfiles1DFreeGS(EqProfiles1D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pprime(self, psinorm=None):
        return self._eq.backend.pprime(psinorm or self.psi_nrom)

    def ffprime(self,  psinorm=None):
        return self._eq.backend.ffprime(psinorm or self.psi_nrom)

    def pressure(self,   psinorm=None):
        return self._eq.backend.pressure(psinorm or self.psi_nrom)

    def fpol(self,   psinorm=None):
        return self._eq.backend.fpol(psinorm or self.psi_nrom)

    def f(self,   psinorm=None):
        return self.psi_nrom

    def dvolume_dpsi(self, psi_norm=None):
        return self.psi_nrom

    def gm2(self, psi_norm=None):
        return self.psi_nrom


class EquilibriumFreeGS(Equilibrium):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def backend(self):
        return self._backend

    def load(self, ids=None, *args,  **kwargs):
        super().load(ids, *args, **kwargs)
        eq_wall = freegs.machine.Wall(self.tokamak.wall.limiter.outline.r(), self.tokamak.wall.limiter.outline.z())

        eq_coils = []

        for coil in self.tokamak.pf_active.coil:
            t_coil = freegs.machine.Coil(
                coil.r+coil.width/2, coil.z+coil.height/2, turns=coil.turns())
            eq_coils.append((coil.name(), t_coil))

        tokamak = freegs.machine.Machine(eq_coils, wall=eq_wall)

        dim1 = self.entry.coordinate_system.grid.dim1()
        dim2 = self.entry.coordinate_system.grid.dim2()

        self._backend = freegs.Equilibrium(
            tokamak=tokamak,
            Rmin=min(dim1), Rmax=max(dim1),
            Zmin=min(dim2), Zmax=max(dim2),
            nx=len(dim1), ny=len(dim2),
            boundary=freegs.boundary.freeBoundaryHagenow)
        return self._backend

    # @property
    # def profiles_1d(self):
    #     if not hasattr(self, "_profiles_1d"):
    #         self._profiles_1d = EqProfiles1DFreeGS(self)
    #     return self._profiles_1d

    @property
    def r(self):
        return self.backend.R

    @property
    def z(self):
        return self.backend.Z

    @property
    def psi(self):
        return self.backend.psi()

    @property
    def psi_norm(self):
        return self.backend.ps

    @property
    def oxpoints(self):
        return freegs.critical.find_critical(self.r, self.z, self.psi)

    def solve(self, core_profiles, fvec=1.0,  constraints=None, **kwargs):
        if isinstance(core_profiles, CoreProfiles):
            psi = self.entry.profiles_1d.psi.__value__()
            profiles = freegs.jtor.ProfilesPprimeFfprime(
                core_profiles.pprime(psi),
                core_profiles.ffprime(psi),
                fvec)
        elif core_profiles is not None and "Ip" in core_profiles:
            profiles = freegs.jtor.jtor.ConstrainPaxisIp(core_profiles["pressure"],  # Plasma pressure on axis [Pascals]
                                                         core_profiles["Ip"],  # Plasma current [Amps]
                                                         fvec)
        else:
            logger.warning(f"Using default profile pressure=1e3 Ip=1e6")
            profiles = freegs.jtor.ConstrainPaxisIp(1e3,  # Plasma pressure on axis [Pascals]
                                                    1e6,  # Plasma current [Amps]
                                                    fvec)

        constrain = freegs.control.constrain(**collections.ChainMap(constraints or {}, kwargs, self.entry.constraints))

        logger.debug(f"Solve Equilibrium [{self.__class__.__name__}] Start")
        freegs.solve(self.backend, profiles, constrain)
        logger.debug(f"Solve Equilibrium [{self.__class__.__name__}] End")

        return
