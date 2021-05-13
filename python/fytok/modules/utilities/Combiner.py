import collections.abc
from functools import cached_property
from typing import Any, Generic, MutableSequence, Sequence

import numpy as np
from spdm.data.Entry import Entry
from spdm.data.Function import Function
from spdm.data.Node import Node
from spdm.util.logger import logger
from spdm.util.utilities import normalize_path, try_get


class Combiner(Entry):
    __slots__ = "_path",

    def __init__(self, cache: Sequence, *args,    **kwargs) -> None:
        super().__init__(cache, *args, **kwargs)

    def get(self, path, *args, **kwargs):
        if len(self._data) == 0:
            logger.warning(f"Combiner of empty list!")
            return None
        path = self._prefix + normalize_path(path)

        if len(path) == 0:
            raise KeyError(f"Empty path!")
        else:
            cache = [try_get(d, path) for d in self._data]

        if all([isinstance(d, (np.ndarray, Function, float, int)) for d in cache]):
            if len(cache) == 1:
                return np.asarray(cache[0])
            elif len(cache) > 1:
                return np.add.reduce([np.asarray(d) for d in cache])
        else:
            return Combiner(self._data, prefix=path)

    def put(self, key, value: Any):
        raise NotImplementedError()

    def iter(self):
        return NotImplemented


def combiner(*args, parent=None, **kwargs):
    return Node(Combiner(*args, **kwargs), parent=parent)
