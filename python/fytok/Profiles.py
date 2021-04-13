import collections
from functools import cached_property

import numpy as np
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.numerical.Function import Function
from spdm.util.logger import logger

from .RadialGrid import RadialGrid


class Profiles(PhysicalGraph):
    def __init__(self, *args, grid: RadialGrid = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._grid = grid

    @property
    def grid(self):
        return self._grid

    def __post_process__(self, value, *args, **kwargs):
        if isinstance(value, (collections.abc.Mapping, collections.abc.MutableSequence)):
            return super().__post_process__(value, *args, **kwargs)
        elif isinstance(value, Function) or (isinstance(value, np.ndarray) and self._grid.rho_tor_norm.shape != value.shape):
            return value
        elif isinstance(value, (int, float, np.ndarray)) or callable(value):
            return Function(self._grid.rho_tor_norm, value)
        else:
            return super().__post_process__(value, *args, **kwargs)
