import collections.abc
from functools import cached_property
from typing import Sequence
import numpy as np
from spdm.data.Function import Function
from spdm.data.Node import Dict, List, Node
from spdm.data.Profiles import Profiles
from spdm.data.TimeSeries import TimeSeries, TimeSlice
from spdm.util.logger import logger

from ..utilities.IDS import IDS, IDSCode
from ..utilities.Misc import Identifier


class Combiner(Node):
    __slots__ = "_prefix", "_axis"

    def __init__(self, prefix=None, axis=None) -> None:
        super().__init__(self)
        self._axis = axis
        if prefix is None:
            self._prefix = []
        elif isinstance(prefix, str):
            self._prefix = prefix.split('.')
        elif not isinstance(prefix, collections.abc.MutableSequence):
            self._prefix = [prefix]

    def __getattr__(self, k):
        path = self._prefix+[k]
        axis = self._axis
        res = None
        for d in self._data:
            v = getattr(d, path, None)
            if v == None:
                continue
            elif isinstance(v, Function):
                if axis is None:
                    axis = v.x
                v = v(axis)
            elif isinstance(v, collections.abc.Mapping):
                raise NotImplementedError()

            if res is None:
                res = v
            else:
                res += v
        if res is None:
            return Combiner(self._cache, path, self._axis)
        else:
            return res
