from functools import cached_property
from typing import Callable, Tuple, TypeVar

import numpy as np
from spdm.data.Function import Function
from spdm.utils.logger import logger

from ..geometry.GeoObject import GeoObject, _TCoord
from .GeoObject import GeoObject
from .Point import Point


class Curve(GeoObject):
    # @staticmethod
    # def __new__(cls, *args, type=None, **kwargs):
    #     if len(args) == 0:
    #         raise RuntimeError(f"Illegal input! {len(args)}")
    #     shape = [(len(a) if isinstance(a, np.ndarray) else 1) for a in args]
    #     if all([s == 1 for s in shape]):
    #         return object.__new__(Point)
    #     elif cls is not Curve:
    #         return object.__new__(cls)
    #     else:
    #         # FIXME：　find module
    #         return object.__new__(Curve)

    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def points(self, *args, **kwargs):
        return super().points(*args, **kwargs)

    @cached_property
    def dl(self) -> np.ndarray:
        x, y = np.moveaxis(self.points(), -1, 0)

        a, b = self.derivative()

        # a = a[:-1]
        # b = b[:-1]
        dx = x[1:]-x[:-1]
        dy = y[1:]-y[:-1]

        m1 = (-a[:-1]*dy+b[:-1]*dx)/(a[:-1]*dx+b[:-1]*dy)

        # a = np.roll(a, 1, axis=0)
        # b = np.roll(b, 1, axis=0)

        m2 = (-a[1:]*dy+b[1:]*dx)/(a[1:]*dx+b[1:]*dy)

        return np.sqrt(dx**2+dy**2)*(1 + (2.0*m1**2+2.0*m2**2-m1*m2)/30)

    @cached_property
    def length(self):
        return np.sum(self.dl)

    def integral(self, func: Callable[[_TCoord, _TCoord], _TCoord]) -> float:
        x, y = self.xyz
        val = func(x, y)
        # c_pts = self.points((self._mesh[0][1:] + self._mesh[0][:-1])*0.5)

        return np.sum(0.5*(val[:-1]+val[1:]) * self.dl)

    # def average(self, func: Callable[[_TCoord, _TCoord], _TCoord]) -> float:
    #     return self.integral(func)/self.length

    def encloses_point(self, *x: float, **kwargs) -> bool:
        return super().enclosed(**x, **kwargs)

    def trim(self):
        return NotImplemented

    def remesh(self, mesh_type=None):
        return NotImplemented


class Line(Curve):
    def __init__(self, *args,   **kwargs) -> None:
        super().__init__(*args, is_closed=False, **kwargs)


def intersect2d(a0: Point, a1: Point, b0: Point, b1: Point) -> Tuple[float, float]:
    da = a1-a0
    db = b1-b0
    dp = a0-b0
    dap = [-da[1], da[0]]
    dbp = [-db[1], db[0]]
    return (np.dot(dbp, dp) / np.dot(dbp, da).astype(float)), (np.dot(dap, dp) / np.dot(dap, db).astype(float))
