from typing import Callable, Iterator, Sequence, Type, Union

from ..geometry.GeoObject import GeoObject
from spdm.util.logger import logger
from .Mesh import Mesh


class StructuredMesh(Mesh):
    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def axis(self, idx, axis=0):
        return NotImplemented

    def remesh(self, *arg, **kwargs):
        return NotImplemented

    def interpolator(self, Z):
        return NotImplemented

    def find_critical_points(self, Z):
        X, Y = self.points
        xmin = X.min()
        xmax = X.max()
        ymin = Y.min()
        ymax = Y.max()
        dx = (xmax-xmin)/X.shape[0]
        dy = (ymax-ymin)/X.shape[1]
        yield from find_critical_points(self.interpolator(Z),  xmin, ymin, xmax, ymax, tolerance=[dx, dy])

    def sub_axis(self, axis=0) -> Iterator[GeoObject]:
        for idx in range(self.shape[axis]):
            yield self.axis(idx, axis=axis)
