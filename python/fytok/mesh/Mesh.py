import collections
from functools import cached_property
from typing import Callable, Iterator, Sequence, Tuple, Type, Union

import numpy as np

from spdm.SpObject import SpObject
from ..geometry.GeoObject import GeoObject
from spdm.logger import logger


class Mesh(SpObject):

    @staticmethod
    def __new__(cls,  mesh=None, *args,   **kwargs):
        if cls is not Mesh:
            return object.__new__(cls)

        n_cls = None
        if mesh is None or mesh == "rectilinear":
            from .RectilinearMesh import RectilinearMesh
            n_cls = RectilinearMesh
        else:
            raise NotImplementedError(mesh)

        return object.__new__(n_cls)

    def __init__(self, mesh=None, *args, ndims=None, uv=None, rank=None, shape=None, name=None, unit=None, cycle=None, **kwargs) -> None:
        self._rank = rank or len(shape or [])
        self._shape = shape or []
        self._ndims = ndims or self._rank
        self._uv = uv

        name = name or [""] * self._ndims
        if isinstance(name, str):
            self._name = name.split(",")
        elif not isinstance(name, collections.abc.Sequence):
            self._name = [name]

        unit = unit or [None] * self._ndims
        if isinstance(unit, str):
            unit = unit.split(",")
        elif not isinstance(unit, collections.abc.Sequence):
            unit = [unit]
        if len(unit) == 1:
            unit = unit * self._ndims
        # self._unit = [*map(Unit(u for u in unit))]

        cycle = cycle or [False] * self._ndims
        if not isinstance(cycle, collections.abc.Sequence):
            cycle = [cycle]
        if len(cycle) == 1:
            cycle = cycle * self._ndims
        self._cycle = cycle

        # logger.debug(f"Create {self.__class__.__name__} rank={self.rank} shape={self.shape} ndims={self.ndims}")

    @property
    def name(self) -> str:
        return self._name

    @property
    def unit(self):
        return self._unit

    @property
    def cycle(self):
        return self._cycle

    @property
    def ndims(self) -> int:
        return self._ndims

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def topology_rank(self):
        return self.ndims

    @cached_property
    def bbox(self) -> Sequence[float]:
        return NotImplemented

    @cached_property
    def dx(self) -> Sequence[float]:
        return NotImplemented

    @cached_property
    def boundary(self):
        return NotImplemented

    @property
    def xy(self) -> Sequence[np.ndarray]:
        return NotImplemented

    def new_dataset(self, *args, **kwargs):
        return np.ndarray(self._shape, *args, **kwargs)

    def interpolator(self, Z):
        return NotImplemented

    def axis(self, *args, **kwargs) -> GeoObject:
        return NotImplemented

    def axis_iter(self, axis=0) -> Iterator[GeoObject]:
        for idx, u in enumerate(self._uv[axis]):
            yield u, self.axis(idx, axis=axis)
