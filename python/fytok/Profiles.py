import collections

import numpy as np
from spdm.data.Group import Group
from spdm.data.Function import Function
from spdm.util.logger import logger
from spdm.data.AttributeTree import AttributeTree


class Profiles(AttributeTree):
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

    def __post_process__(self, d, *args, parent=None, **kwargs):
        if isinstance(d, Function):
            return d
        elif isinstance(d, (int, float, np.ndarray)) or callable(d):
            return Function(self._axis, d)
        elif d is None or d == None:
            return Function(self._axis, 0.0)
        else:
            return super().__post_process__(d, *args, parent=parent,  **kwargs)
