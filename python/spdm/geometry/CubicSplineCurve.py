
import collections
from functools import cached_property
from typing import Union

import numpy as np
from scipy import constants
from scipy.interpolate import CubicSpline, PPoly

from spdm.utils.logger import logger
from .Curve import Curve

TWOPI = 2.0*constants.pi


class CubicSplineCurve(Curve):
    def __init__(self,   *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # if not isinstance(self._mesh, collections.abc.Sequence):
        #     raise NotImplementedError(type(self._mesh))
    def points(self,  *args, **kwargs) -> np.ndarray:
        if len(args) == 0:
            return self._points
        else:
            return self._spl(*args, **kwargs)

    @cached_property
    def _spl(self) -> PPoly:
        if self._mesh is None:
            self._mesh = [np.linspace(0, 1, len(self._points))]
        return CubicSpline(self._mesh[0], self._points, bc_type="periodic" if self.is_closed else "not-a-knot")

    @cached_property
    def _derivative(self):
        return self._spl.derivative()

    def derivative(self, *args, **kwargs):
        if len(args) == 0:
            args = self.mesh
        res = self._derivative(*args, **kwargs)
        return res[:, 0], res[:, 1]
