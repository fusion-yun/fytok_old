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

    def refresh(self, *args, core_profiles: CoreProfiles = None, constraints: Equilibrium.Constraints = None, **kwargs):
        logger.debug(f"Dummy actor:  Nothing to do!")


__SP_EXPORT__ = EquilibriumDummy
