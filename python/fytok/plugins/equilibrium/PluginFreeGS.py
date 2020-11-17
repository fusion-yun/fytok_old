import collections
import sys
from functools import cached_property
import matplotlib.pyplot as plt
import freegs
from freegs.critical import find_separatrix
import numpy as np
from spdm.util.Interpolate import derivate, integral, interpolate, Interpolate2D, Interpolate1D
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles
from spdm.util.AttributeTree import _next_, AttributeTree
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

        self._machine = freegs.machine.Machine(eq_coils, wall=eq_wall)
        self._eq = None

    def _solve(self, profiles=None, constraints=None,  **kwargs):
        if self._eq is None:
            self._eq = freegs.Equilibrium(
                tokamak=self._machine,
                Rmin=min(self.profiles_2d.grid.dim1), Rmax=max(self.profiles_2d.grid.dim1),
                Zmin=min(self.profiles_2d.grid.dim2), Zmax=max(self.profiles_2d.grid.dim2),
                nx=len(self.profiles_2d.grid.dim1), ny=len(self.profiles_2d.grid.dim2),
                psi=self.profiles_2d.psi,
                current=self.global_quantities.ip or 0.0,
                boundary=freegs.boundary.freeBoundaryHagenow)

        if profiles is None:
            profiles = {}
        elif not isinstance(profiles, collections.abc.Mapping):
            raise NotImplementedError()

        fvec = self.vacuum_toroidal_field.b0 * self.vacuum_toroidal_field.r0

        if "pprime" in profiles and "ffprime" in profiles:

            psi_norm = profiles.get("psi_norm", self.profiles_1d.psi_norm)
            pprime = Interpolate1D(profiles["pprime"], psi_norm)
            ffprime = Interpolate1D(profiles["ffprime"], psi_norm)

            profiles = freegs.jtor.ProfilesPprimeFfprime(pprime, ffprime, fvec)
            logger.debug("Create Profile:     Specified profile functions p'(psi), ff'(psi)")
        elif "betap" in profiles:
            #  Poloidal beta
            betap = profiles.get("betap", 1.0)
            # Plasma current [Amps]
            ip = profiles.get("ip", 1.0)
            profiles = freegs.jtor.ConstrainBetapIp(profiles["betap"], profiles["ip"], fvec)

            logger.debug(f"""Create Profile: Constrain poloidal Beta and plasma current
                     Plasma pressure on axis    ={pressure} [Pascals],
                     Plasma current Ip          ={ip} [Amps],
                     fvec                       ={fvec} [T.m]""")
        else:
            # Plasma pressure on axis [Pascals]
            pressure = profiles.get("pressure", 1.0e3)
            # Plasma current [Amps]
            ip = profiles.get("ip", 1.0e6)
            Raxis = self.vacuum_toroidal_field.r0
            profiles = freegs.jtor.ConstrainPaxisIp(pressure, ip, fvec, Raxis=Raxis)

            logger.debug(f"""Create Profile: Constrain pressure on axis and plasma current
                     Plasma pressure on axis    ={pressure} [Pascals],
                     Plasma current Ip          ={ip} [Amps],
                     fvec                       ={fvec} [T.m],
                     Raxis                      ={Raxis} [m]
                     """)
        if constraints is None:
            constraints = {}
        psivals = constraints.get("psivals", None) or self.constraints.psivals or []
        isoflux = constraints.get("isoflux", None) or self.constraints.isoflux or []
        xpoints = constraints.get("xpoints", None) or self.constraints.xpoints or []

        constraints = freegs.control.constrain(psivals=psivals, isoflux=isoflux, xpoints=xpoints)

        try:
            freegs.solve(self._eq, profiles, constraints)
        except ValueError as error:
            raise RuntimeError(f"Solve G-S equation failed [{self.__class__.__name__}]! {error}")

        self._update_cache(self._eq)

    def _update_cache(self, eq, ntheta=128):
        psi_norm = self.profiles_1d.psi_norm
        self._cache.profiles_1d.pprime = eq.pprime(psi_norm)
        self._cache.profiles_1d.f_df_dpsi = eq.ffprime(psi_norm)
        self._cache.profiles_1d.f = eq.fpol(psi_norm)
        self._cache.profiles_1d.pressure = eq.pressure(psi_norm)
        self._cache.profiles_1d.q = eq.q(psi_norm)

        r, z = np.meshgrid(self.profiles_2d.grid.dim1, self.profiles_2d.grid.dim2, indexing="ij")
        self._cache.profiles_2d.r = r
        self._cache.profiles_2d.z = z
        self._cache.profiles_2d.psi = eq.psiRZ(r, z)
        self._cache.global_quantities.beta_pol = eq.poloidalBeta()
        self._cache.global_quantities.ip = eq.plasmaCurrent()
        self._cache.global_quantities.volume = eq.plasmaVolume()

        self._cache.boundary.psi = eq.psi_bndry

        brdy = eq.separatrix(ntheta)

        self._cache.boundary.outline.r = brdy[:, 0]
        self._cache.boundary.outline.z = brdy[:, 1]

    # @property
    # def critical_points(self):
    #     R = self.profiles_2d.r
    #     Z = self.profiles_2d.z
    #     psi = self.profiles_2d.psi

    #     # find magnetic axis (o-point) and x-points
    #     opt, xpt = freegs.critical.find_critical(R, Z, psi)

    #     if not opt or not xpt:
    #         raise RuntimeError(f"Can not find O-point or X-points!")
    #     return opt, xpt
