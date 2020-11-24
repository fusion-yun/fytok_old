import collections
import math
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from spdm.util.AttributeTree import AttributeTree, _last_, _next_
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from fytok.Tokamak import Tokamak
from fytok.PulseSchedule import PulseSchedule


class Scenario(AttributeTree):
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
