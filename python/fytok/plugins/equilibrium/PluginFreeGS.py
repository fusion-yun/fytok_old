import collections
import sys
from functools import cached_property

import freegs
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from spdm.util.AttributeTree import AttributeTree, _next_
from spdm.util.Interpolate import (Interpolate1D, Interpolate2D, derivate,
                                   integral, interpolate)
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles

from ...CoreProfiles import CoreProfiles
from ...Equilibrium import Equilibrium


def is_none(v):
    return (not v) or v is NotImplemented


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
        self._backend = None

    def _solve(self, profiles=None, constraints=None,  **kwargs):
        if self._backend is None:
            if "psi" in kwargs:
                psi = kwargs["psi"]
            else:
                psi = self.profiles_2d.psi
            if "current" in kwargs:
                current = kwargs["current"]
            else:
                current = self.global_quantities.ip or 0.0

            self._backend = freegs.Equilibrium(
                tokamak=self._machine,
                Rmin=min(self.profiles_2d.grid.dim1), Rmax=max(self.profiles_2d.grid.dim1),
                Zmin=min(self.profiles_2d.grid.dim2), Zmax=max(self.profiles_2d.grid.dim2),
                nx=len(self.profiles_2d.grid.dim1), ny=len(self.profiles_2d.grid.dim2),
                psi=psi,
                current=current,
                boundary=freegs.boundary.freeBoundaryHagenow)

        if profiles is None:
            profiles = {}
        elif not isinstance(profiles, collections.abc.Mapping):
            raise NotImplementedError()

        fvac = self.vacuum_toroidal_field.b0 * self.vacuum_toroidal_field.r0

        pprime = profiles.get("pprime", None)
        ffprime = profiles.get("ffprime", None)

        #  Poloidal beta
        betap = profiles.get("betap", None) or self.global_quantities.beta_pol
        # Plasma current [Amps]
        ip = profiles.get("ip", None) or self.global_quantities.ip

        if not is_none(pprime) and not is_none(ffprime):
            profiles = freegs.jtor.ProfilesPprimeFfprime(pprime, ffprime, fvac)
            logger.debug("Create Profile: Specified profile functions p'(psi), ff'(psi)")
        elif not is_none(ip) and not is_none(betap):
            profiles = freegs.jtor.ConstrainBetapIp(betap, ip, fvac)

            logger.debug(f"""Create Profile: Constrain poloidal Beta and plasma current
                     Betap                      ={betap} [-],
                     Plasma current Ip          ={ip} [Amps],
                     fvec                       ={fvac} [T.m]""")
        else:
            # Plasma pressure on axis [Pascals]
            pressure = profiles.get("pressure", None) or  \
                self.global_quantities.beta_tor*(self.vacuum_toroidal_field.b0**2)/(2.0*scipy.constants.mu_0)
            logger.debug(self.global_quantities.cache)
            if is_none(pressure):
                raise RuntimeError(f"pressure is not defined!")
            logger.debug((pressure, ip, fvac))

            Raxis = self.vacuum_toroidal_field.r0
            profiles = freegs.jtor.ConstrainPaxisIp(pressure, ip, fvac, Raxis=Raxis)

            logger.debug(f"""Create Profile: Constrain pressure on axis and plasma current
                     Plasma pressure on axis    ={pressure} [Pascals],
                     Plasma current Ip          ={ip} [Amps],
                     fvec                       ={fvac} [T.m],
                     Raxis                      ={Raxis} [m]
                     """)
        if constraints is None:
            constraints = {}
        psivals = constraints.get("psivals", None) or self.constraints.psivals or []
        isoflux = constraints.get("isoflux", None) or self.constraints.isoflux or []
        xpoints = constraints.get("xpoints", None) or self.constraints.xpoints or []

        constraints = freegs.control.constrain(psivals=psivals, isoflux=isoflux, xpoints=xpoints)

        try:
            freegs.solve(self._backend, profiles, constraints)
        except ValueError as error:
            logger.error(f"Solve G-S equation failed [{self.__class__.__name__}]! {error}")

    def update_cache(self):

        psi_norm = self.profiles_1d.psi_norm
        super().update_cache()
        self.cache.profiles_1d.pprime = self._backend.pprime(psi_norm)
        self.cache.profiles_1d.f_df_dpsi = self._backend.ffprime(psi_norm)
        self.cache.profiles_1d.f = self._backend.fpol(psi_norm)
        self.cache.profiles_1d.pressure = self._backend.pressure(psi_norm)
        self.cache.profiles_1d.q = self._backend.q(psi_norm)

        self.cache.profiles_2d.psi = self._backend.psiRZ(self.profiles_2d.r, self.profiles_2d.z)

        self.cache.global_quantities.beta_pol = self._backend.poloidalBeta()
        self.cache.global_quantities.ip = self._backend.plasmaCurrent()

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
