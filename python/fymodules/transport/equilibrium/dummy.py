import collections

import matplotlib.pyplot as plt
from fytok.device.PFActive import PFActive
from fytok.device.Wall import Wall
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.data.Node import Dict
from spdm.numlib import constants, np
from spdm.util.logger import logger


class EquilibriumDummy(Equilibrium):

    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)

    def refresh(self, *args, time=None,  **kwargs) -> float:
        residual = super().refresh(time=time)
        if len(args) > 0:
            self.update(args[0])
        if len(kwargs) > 0:
            self.update(kwargs)
        return residual


__SP_EXPORT__ = EquilibriumDummy
