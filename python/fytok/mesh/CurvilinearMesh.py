import collections
from functools import cached_property, lru_cache
from math import log
from typing import Iterator, Sequence, Type, Union

import numpy as np
from scipy import interpolate
from scipy.ndimage.interpolation import geometric_transform
from spdm.logger import logger
from spdm.util.utilities import convert_to_named_tuple

from ..geometry.BSplineSurface import BSplineSurface
from ..geometry.CubicSplineCurve import CubicSplineCurve
from ..geometry.Curve import Curve
from ..geometry.GeoObject import GeoObject
from ..geometry.Point import Point
from .Mesh import Mesh
from .StructuredMesh import StructuredMesh


class CurvilinearMesh(StructuredMesh):
    """
        A `curvilinear grid` or `structured grid` is a grid with the same combinatorial structure as a regular grid,
        in which the cells are quadrilaterals or [general] cuboids, rather than rectangles or rectangular cuboids.
            -- [https://en.wikipedia.org/wiki/Regular_grid]
    """
    TOLERANCE = 1.0e-5

    def __init__(self, geo_mesh: Union[np.ndarray, Sequence[GeoObject], GeoObject], uv=None,  *args,   ** kwargs) -> None:
        rank = len(uv)
        shape = [len(d) for d in uv]
        if isinstance(geo_mesh, np.ndarray):
            if geo_mesh.shape[:-1] != tuple(shape):
                raise ValueError(
                    f"Illegal shape!  {geo_mesh.shape[:-1]} != {tuple(shape)}")
            ndims = geo_mesh.shape[-1]
            xy = geo_mesh
            surf = None
            raise NotImplementedError(f"NOT COMPLETE! xy -> surface")

        elif isinstance(geo_mesh, collections.abc.Sequence) and isinstance(geo_mesh[0], GeoObject):
            ndims = geo_mesh[0].ndims
            if len(uv[0]) != len(geo_mesh):
                raise ValueError(
                    f"Illegal number of sub-surface {len(self[uv[0]])} != {len(geo_mesh)}")
            surf = geo_mesh
        elif isinstance(geo_mesh, GeoObject):
            raise NotImplementedError(type(geo_mesh))
        else:
            raise TypeError(
                f"geo_mesh should be np.ndarray, Sequence[GeoObject] or GeoObject, not {type(geo_mesh)}")

        super().__init__(*args, uv=np.asarray(uv),
                         shape=shape, rank=rank, ndims=ndims, **kwargs)
        self._sub_surf = surf

    def axis(self, idx, axis=0):
        if axis == 0:
            return self._sub_surf[idx]
        else:
            s = [slice(None, None, None)]*self.ndims
            s[axis] = idx
            s = s+[slice(None, None, None)]

            sub_xy = self.xy[tuple(s)]  # [p[tuple(s)] for p in self._xy]
            sub_uv = [self._uv[(axis+i) % self.ndims]
                      for i in range(1, self.ndims)]
            sub_cycle = [self.cycle[(axis+i) % self.ndims]
                         for i in range(1, self.ndims)]

            return CurvilinearMesh(sub_xy, sub_uv,  cycle=sub_cycle)

    @property
    def uv(self) -> np.ndarray:
        return self._uv

    @cached_property
    def xy(self) -> np.ndarray:
        return np.stack([surf.points(self.uv[1]) for idx, surf in enumerate(self._sub_surf)], axis=0)

    # def pushforward(self, new_uv):
    #     new_shape = [len(u) for u in new_uv]
    #     if new_shape != self.shape:
    #         raise ValueError(f"illegal shape! {new_shape}!={self.shape}")
    #     return CurvilinearMesh(self._xy, new_uv, cycle=self.cycle)

    def interpolator(self, value,  **kwargs):
        if value.shape != self.shape:
            raise ValueError(f"{value.shape} {self.shape}")

        if self.ndims == 1:
            interp = interpolate.InterpolatedUnivariateSpline(
                self._dims[0], value,  **kwargs)
        elif self.ndims == 2:
            interp = interpolate.RectBivariateSpline(
                self._dims[0], self._dims[1], value, ** kwargs)
        else:
            raise NotImplementedError(f"NDIMS {self.ndims}>2")
        return interp

    @cached_property
    def boundary(self):
        return convert_to_named_tuple({"inner": self.axis(0, 0),  "outer": self.axis(-1, 0)})

    @cached_property
    def geo_object(self):
        if self.rank == 1:
            if all([np.var(x)/np.mean(x**2) < CurvilinearMesh.TOLERANCE for x in self.xy.T]):
                gobj = Point(*[x[0] for x in self.xy.T])
            else:
                gobj = CubicSplineCurve(
                    self.xy, self._uv[0], is_closed=self.cycle[0])
        elif self.rank == 2:
            gobj = BSplineSurface(self.xy, self._uv,  is_closed=self.cycle)
        else:
            raise NotImplementedError()

        return gobj
