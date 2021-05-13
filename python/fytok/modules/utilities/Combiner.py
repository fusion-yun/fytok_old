import collections.abc
import enum
from functools import cached_property
from typing import Any, Generic, MutableSequence, Sequence

import numpy as np
from spdm.data.Entry import Entry
from spdm.data.Function import Function
from spdm.data.Node import Node
from spdm.util.logger import logger
from spdm.util.utilities import normalize_path, try_get


class Combiner(Entry):
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

        cache = [d for d in cache if isinstance(d, (np.ndarray, Function, float, int))]

        if len(cache) == 0:
            return Combiner(self._data, prefix=path)
        elif len(cache) == 1:
            return (cache[0])
        else:
            return np.add.reduce(cache)

    def put(self, key, value: Any):
        raise NotImplementedError()

    def iter(self):
        cache = [try_get(d, self._prefix).__iter__() for d in self._data]
        if len(cache) == 0:
            return NotImplementedError()
        for d in cache[0]:
            yield Combiner([d, *[next(it) for it in cache[1:]]])


def combiner(*args, parent=None, **kwargs):
    return Node(Combiner(*args, **kwargs), parent=parent)
