import collections

from fytok.device.PFActive import PFActive
from fytok.device.Wall import Wall
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.Equilibrium import Equilibrium
from scipy import constants
from spdm.util.logger import logger
from spdm.data import Dict, Function


class EquilibriumDummy(Equilibrium):

    def refresh(self, *args, time=None,  **kwargs) -> float:
        residual = super().refresh(time=time)
        if len(args) > 0:
            self.update(args[0])
        if len(kwargs) > 0:
            self.update(kwargs)
        return residual


__SP_EXPORT__ = EquilibriumDummy
