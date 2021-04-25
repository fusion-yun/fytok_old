import collections
from functools import cached_property

import numpy as np
from numpy.lib.arraysetops import isin
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.numerical.Function import Function
from spdm.util.logger import logger


class Profiles(PhysicalGraph):
    def __init__(self,   *args, axis=None, ** kwargs):
        super().__init__(*args, **kwargs)
        if axis is None:
            self._axis = np.linspace(0, 1.0, 128)
        elif isinstance(axis, int):
            self._axis = np.linspace(0, 1.0, axis)
        elif isinstance(axis, np.ndarray):
            self._axis = axis.view(np.ndarray)
        else:
            raise TypeError(type(axis))

    @property
    def axis(self):
        return self._axis

    def __post_process__(self, value, *args, **kwargs):
        if isinstance(value, (collections.abc.Mapping, collections.abc.MutableSequence)):
            return super().__post_process__(value, *args, **kwargs)
        elif isinstance(value, Function) or (isinstance(value, np.ndarray) and self._axis.shape != value.shape):
            return value
        elif isinstance(value, (int, float, np.ndarray)) or callable(value):
            return Function(self._axis, value)
        else:
            return super().__post_process__(value, *args, **kwargs)
