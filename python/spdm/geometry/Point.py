from __future__ import print_function

import collections.abc
from functools import cached_property
from typing import Callable, Collection, TypeVar

import numpy as np
from spdm.utils.logger import logger

from ..geometry.GeoObject import GeoObject, _TCoord
from .GeoObject import GeoObject


class Point(GeoObject):
    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1 and isinstance(args[0], collections.abc.Sequence):
            args = args[0]
        super().__init__(args, **kwargs)

    def points(self, *uv, **kwargs) -> np.ndarray:
        """
        """
        if len(uv) == 0:
            return super().points()
        else:
            return np.asarray([self._points]*len(uv[0]))

    def __call__(self, *args, **kwargs):
        return self._x

    def map(self,  *args, **kwargs):
        return self._x

    @property
    def dl(self):
        return 0.0

    @property
    def length(self):
        return 0

    def integral(self, func: Callable[[_TCoord, _TCoord], _TCoord]) -> float:
        return func(*self._points)

    def average(self, func: Callable[[_TCoord, _TCoord], _TCoord]) -> float:
        return func(*self._points)
