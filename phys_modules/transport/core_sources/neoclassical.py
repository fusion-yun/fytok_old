
import collections

import numpy as np
import scipy.constants
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreSources import CoreSources
from fytok.modules.transport.Equilibrium import Equilibrium
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Combiner import Combiner
from spdm.data.Function import Function
from spdm.data.Node import Dict, List, Node
from spdm.data.Profiles import Profiles
from spdm.data.TimeSeries import TimeSequence, TimeSeries, TimeSlice
from spdm.flow.Actor import Actor, ActorBundle
from spdm.util.logger import logger


class NeoClassical(CoreSources.Source):
    def __init__(self, d=None, *args,  **kwargs):
        super().__init__(collections.ChainMap({
            "identifier": {
                "name": f"neoclassical",
                "index": 5,
                "description": f"{self.__class__.__name__}  Neoclassical model, based on  Tokamaks, 3ed, J.A.Wesson 2003"
            }}, d or {}), *args, **kwargs)

    def update(self, *args,
               equilibrium: Equilibrium.TimeSlice,
               core_profiles: CoreProfiles.TimeSlice,
               **kwargs):
        return 0.0
