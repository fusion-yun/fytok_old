from functools import cached_property, lru_cache
from typing import Sequence, Tuple

import numpy as np
from scipy.interpolate import interpolate

from ..geometry.Curve import Line
from ..geometry.Point import Point
from spdm.utils.logger import logger
from .StructuredMesh import StructuredMesh


class RectilinearMesh(StructuredMesh):
    """
        A `rectilinear grid` is a tessellation by rectangles or rectangular cuboids (also known as rectangular parallelepipeds)
        that are not, in general, all congruent to each other. The cells may still be indexed by integers as above, but the
        mapping from indexes to vertex coordinates is less uniform than in a regular grid. An example of a rectilinear grid
        that is not regular appears on logarithmic scale graph paper.
            -- [https://en.wikipedia.org/wiki/Regular_grid]

    """

    def __init__(self, desc=None, *args,  **kwargs) -> None:
        def normalize_dim(d):
            if isinstance(d, np.ndarray):
                return d
            elif isinstance(d, int):
                return np.linspace(0, 1, d)
            else:
                raise TypeError(d)

        self._dims = [normalize_dim(d) for d in args]
        super().__init__(*args, shape=[len(d) for d in self._dims],  **kwargs)

    @cached_property
    def bbox(self) -> Sequence[float]:
        return [*[d[0] for d in self._dims], *[d[-1] for d in self._dims]]

    @cached_property
    def dx(self) -> Sequence[float]:
        return [(d[-1]-d[0])/self.shape[idx] for idx, d in enumerate(self._dims)]

    def axis(self, idx, axis=0):
        p0 = [d[0] for d in self._dims]
        p1 = [d[-1] for d in self._dims]
        p0[axis] = self._dims[axis][idx]
        p1[axis] = self._dims[axis][idx]

        try:
            res = Line(p0, p1, is_closed=self.cycle[axis])
        except ValueError as error:
            res = Point(*p0)
        return res

    @cached_property
    def xy(self) -> Sequence[np.ndarray]:
        if self.ndims == 1:
            return [self._dims[0]]
        elif self.ndims == 2:
            return np.meshgrid(*self._dims, indexing="ij")
        else:
            raise NotImplementedError()

    def point(self, *idx):
        return [d[idx[s]] for s, d in enumerate(self._dims)]

    def interpolator(self, value,  **kwargs):
        if value.shape != self.shape:
            raise ValueError(f"{value.shape} {self.shape}")

        if self.ndims == 1:
            interp = interpolate.InterpolatedUnivariateSpline(self._dims[0], value,  **kwargs)
        elif self.ndims == 2:
            interp = interpolate.RectBivariateSpline(self._dims[0], self._dims[1], value,  **kwargs)
        else:
            raise NotImplementedError(f"NDIMS {self.ndims}>2")

        return interp

    @ cached_property
    def dl(self):
        dX = (np.roll(self.points[0], 1, axis=1) - np.roll(self.points[0], -1, axis=1))/2.0
        dY = (np.roll(self.points[1], 1, axis=1) - np.roll(self.points[1], -1, axis=1))/2.0
        return dX, dY
