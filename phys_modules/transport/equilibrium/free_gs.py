import collections
import pprint
import sys
from  functools import cached_property

import freegs
import matplotlib.pyplot as plt
from spdm.numlib import np
from spdm.numlib import constants
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.data.Node import Dict
from spdm.util.logger import logger
from fytok.device.Wall import Wall
from fytok.device.PFActive import PFActive


def is_none(v):
    return (not v) or v is NotImplemented


class EquilibriumFreeGS(Equilibrium):

    def __init__(self, *args, wall: Wall, pf_active: PFActive = None, **kwargs):
        super().__init__(*args, **kwargs)

        # if wall is None:
        #     wall = self._parent.wall
        # if pf_active:
        #     pf_active = self._parent.pf_active

        eq_wall = freegs.machine.Wall(wall.description_2d.limiter.unit.outline.r,
                                      wall.description_2d.limiter.unit.outline.z)

        eq_coils = []

        for coil in pf_active.coil:

            rect = coil.element.geometry.rectangle
            turns = coil.element.turns_with_sign

            t_coil = freegs.machine.Coil(
                rect.r+rect.width/2,
                rect.z+rect.height/2,
                turns=turns)
            eq_coils.append((coil.name, t_coil))

        self._machine = freegs.machine.Machine(eq_coils, wall=eq_wall)
        self._backend = None

    def update(self, profiles=None, constraints=None, Ip=None, fvac=None, B0=None, **kwargs):

        if not hasattr(self, "_backend") or self._backend is None:
            self._backend = freegs.Equilibrium(
                tokamak=self._machine,
                Rmin=min(self.profiles_2d.grid.dim1), Rmax=max(self.profiles_2d.grid.dim1),
                Zmin=min(self.profiles_2d.grid.dim2), Zmax=max(self.profiles_2d.grid.dim2),
                nx=len(self.profiles_2d.grid.dim1), ny=len(self.profiles_2d.grid.dim2),
                # psi=self.profiles_2d.psi,
                current=self.global_quantities.ip or 0.0,
                boundary=freegs.boundary.freeBoundaryHagenow)

        if profiles is None:
            profiles = {}
        elif not isinstance(profiles, collections.abc.Mapping):
            raise NotImplementedError()

        B0 = B0 or self.vacuum_toroidal_field.b0

        fvac = fvac or B0 * self.vacuum_toroidal_field.r0

        pprime = profiles.get("pprime", None)
        ffprime = profiles.get("ffprime", None)

        #  Poloidal beta
        betap = profiles.get("betap", None) or self.global_quantities.beta_pol
        # Plasma current [Amps]
        Ip = Ip or profiles.get("ip", None) or self.global_quantities.ip

        if not is_none(pprime) and not is_none(ffprime):
            profiles = freegs.jtor.ProfilesPprimeFfprime(pprime, ffprime, fvac)
            logger.debug("Create Profile: Specified profile functions p'(psi), ff'(psi)")
        elif not is_none(Ip) and not is_none(betap):
            profiles = freegs.jtor.ConstrainBetapIp(betap, Ip, fvac)

            logger.debug(f"""Create Profile: Constrain poloidal Beta and plasma current
                     Betap                      ={betap} [-],
                     Plasma current Ip          ={Ip} [Amps],
                     fvac                       ={fvac} [T.m]""")
        else:
            # Plasma pressure on axis [Pascals]
            pressure = profiles.get("pressure", None) or  \
                self.global_quantities.beta_tor*(self.vacuum_toroidal_field.b0**2)/(2.0*constants.mu_0)
            if is_none(pressure):
                raise RuntimeError(f"pressure is not defined!")

            Raxis = self.vacuum_toroidal_field.r0
            profiles = freegs.jtor.ConstrainPaxisIp(pressure, Ip, fvac, Raxis=Raxis)

            logger.debug(f"""Create Profile: Constrain pressure on axis and plasma current
                     Plasma pressure on axis    ={pressure} [Pascals],
                     Plasma current Ip          ={Ip} [Amps],
                     fvac                       ={fvac} [T.m],
                     Raxis                      ={Raxis} [m]
                     """)

        if constraints is None:
            constraints = self.constraints
        else:
            constraints = Dict(constraints)

        psivals = constraints.psivals or []
        isoflux = constraints.isoflux or []
        xpoints = constraints.xpoints or []

        constraints = freegs.control.constrain(
            psivals=psivals,
            isoflux=isoflux,
            xpoints=xpoints
        )

        try:
            freegs.solve(self._backend, profiles, constraints)
        except ValueError as error:
            logger.error(f"Solve G-S equation failed [{self.__class__.__name__}]! {error}")
        else:
            # logger.debug(f"Solve G-S equation Done")
            # super().update()
            self.profiles_2d.update(solver="FreeGS", psi=self._backend.psiRZ)
            self._parent.pf_active.update(self._backend.tokamak.coils)

    def update_cache(self):
        psi_norm = self.profiles_1d.psi_norm
        # super().update_cache()
        if hasattr(self._backend, "_profiles"):
            self.profiles_1d.dpressure_dpsi = self._backend.pprime(psi_norm)
            self.profiles_1d.f_df_dpsi = self._backend.ffprime(psi_norm)
            self.profiles_1d.f = self._backend.fpol(psi_norm)
            self.profiles_1d.pressure = self._backend.pressure(psi_norm)
            try:
                x, q = self._backend.q()
            except Exception:
                pass
            else:
                self.profiles_1d.q = Function(x, q)
                self.global_quantities.beta_pol = self._backend.poloidalBeta()
            self.global_quantities.ip = self._backend.plasmaCurrent()

            self.profiles_2d.update(solver="FreeGS", psi=self._backend.psiRZ)

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


__SP_EXPORT__ = EquilibriumFreeGS
