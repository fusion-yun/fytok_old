import collections
import sys

import freegs
from spdm.util.logger import logger

from ...Equilibrium import Equilibrium


class EquilibriumFreeGS(Equilibrium):
    def __init__(self,  *args, grid=[129, 129], **kwargs):
        super().__init__(*args, **kwargs)
        self._eq = None
        self._grid = grid

    @property
    def equilibrium(self):
        if self._eq is not None:
            return self._eq

        eq_wall = freegs.machine.Wall(self.tokamak.wall.limiter.r,
                                      self.tokamak.wall.limiter.z)
        eq_coils = []

        for name, coil in self.tokamak.pf_coils:
            eq_coils.append((name, freegs.machine.Coil(
                coil.r+coil.width/2, coil.z+coil.height/2, turns=coil.turns)))

        box = [[min(self.tokamak.wall.limiter.r), min(self.tokamak.wall.limiter.z)],
               [max(self.tokamak.wall.limiter.r), max(self.tokamak.wall.limiter.z)]]

        self._eq = freegs.Equilibrium(tokamak=freegs.machine.Machine(eq_coils, eq_wall),
                                      Rmin=box[0][0], Rmax=box[1][0],
                                      Zmin=box[0][1], Zmax=box[1][1],
                                      nx=self._grid[0], ny=self._grid[1],
                                      boundary=freegs.boundary.freeBoundaryHagenow)
        return self._eq

    def plot(self, axis=None, **kwargs):
        return self.equilibrium.plot(axis=axis)

    def solve(self, profiles=None, constrain=None, fvec=1.0, **kwargs):

        if profiles is None:
            profiles = freegs.jtor.ConstrainPaxisIp(1e3,  # Plasma pressure on axis [Pascals]
                                                    1e6,  # Plasma current [Amps]
                                                    fvec)
        elif not isinstance(profiles, collections.abc.Mapping):
            pass
        elif "pprime" in profiles and "ffprime" in profiles:
            profiles = freegs.jtor.ProfilesPprimeFfprime(
                pprime=profiles["pprime"],
                ffprime=profiles["ffprime"],
                fvec=fvec)
        elif "Ip" in profiles:
            profiles = freegs.jtor.jtor.ConstrainPaxisIp(profiles["pressure"],  # Plasma pressure on axis [Pascals]
                                                         profiles["Ip"],  # Plasma current [Amps]
                                                         fvec)
        else:
            raise RuntimeError(f"Illegal profiles type!")

        if constrain is None:
            constrain = kwargs

        if isinstance(constrain, collections.abc.Mapping):
            constrain = freegs.control.constrain(**constrain)
        else:
            raise RuntimeError(f"Illegal constrain type!")

        logger.debug(f"Solve Equilibrium [{self.__class__.__name__}] Start")
        freegs.solve(self.equilibrium, profiles, constrain)
        logger.debug(f"Solve Equilibrium [{self.__class__.__name__}] End")

        return
