import collections
import math
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.Graph import Graph
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger

from fytok.PulseSchedule import PulseSchedule
from fytok.Tokamak import Tokamak


class Scenario(Graph):
    """Scenario

    """

    def __init__(self,  cache=None,  *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__["_cache"] = cache

    @cached_property
    def tokamak(self):
        return Tokamak(self._cache.tokamak)

    @cached_property
    def pulse_schedule(self):
        return PulseSchedule(self._cache.pulse_schedule, tokamak=self.tokamak)
