import collections.abc
from functools import cached_property
from typing import Callable, Sequence, TypeVar, Union

import numpy as np

_TMesh = TypeVar("_TMesh")

_TCoord = TypeVar("_TCoord", float, np.ndarray)


class GeoObject:
    def __init__(self, points: np.ndarray, uv_mesh: Union[Sequence[np.ndarray], _TMesh] = None, /,   **kwargs) -> None:
        """
            points  : coordinates of vertices on mesh, shape= [*shape_of_mesh,ndims]
            uv_mesh : 
                structured :  N ordered sequences whose values are between 0 and 1.
                unstructured mesh: _TMesh map(*uv)-> index
        """
        if not isinstance(points, np.ndarray):
            points = np.asarray(points)

        self._points = points

        if uv_mesh is None:
            uv_mesh = [np.linspace(0, 1, n) for n in self._points.shape[:-1]]
        elif isinstance(uv_mesh, collections.abc.Sequence):
            uv_mesh = [(np.linspace(0, 1, d) if isinstance(d, int) else d) for d in uv_mesh]

        # if isinstance(uv_mesh, collections.abc.Sequence):
        #     uv_mesh = [(d-d.min())/(d.max()-d.min()) for d in uv_mesh]
        self._mesh = uv_mesh

    @property
    def rank(self) -> int:
        """
            0: point
            1: curve
            2: surface
            3: volume
            >=4: not defined
        """
        if isinstance(self._mesh, collections.abc.Sequence):
            return len(self._mesh)
        else:
            return getattr(self._mesh, "ndims", NotImplemented)

    @property
    def shape(self) -> Sequence[int]:
        """
            shape of uv mesh
        """
        if isinstance(self._mesh, collections.abc.Sequence):
            return [len(d) for d in self._mesh]
        else:
            return getattr(self._mesh, "shape", NotImplemented)

    @property
    def ndims(self) -> int:
        """
            dimension of space 
        """
        return self._points.shape[-1]

    @property
    def mesh(self) -> Union[Sequence[np.ndarray], _TMesh]:
        return self._mesh

    def points(self, *uv, **kwargs) -> np.ndarray:
        """
            coordinates of vertices on mesh
            Return: array-like
                shape = [*shape_of_mesh,ndims]
        """
        if len(uv) == 0:
            return self._points
        else:
            raise NotImplementedError(f"{self.__class__.__name__}")

    @cached_property
    def xyz(self) -> np.ndarray:
        """
            coordinates of vertices on mesh
            Return: array-like
                shape = [ndims, *shape_of_mesh]
        """
        return np.moveaxis(self.points(), -1, 0)

    def dl(self, uv=None) -> np.ndarray:
        """
            derivative of shape
            Returns:
                rank==0 : 0
                rank==1 : dl (shape=[n-1])
                rank==2 : dx (shape=[n-1,m-1]), dy (shape=[n-1,m-1])
                rank==3 : dx (shape=[n-1,m-1,l-1]), dy (shape=[n-1,m-1,l-1]), dz (shape=[n-1,m-1,l-1])
        """
        return np.asarray(0)

    @cached_property
    def length(self):
        return np.sum(self.dl)

    def integral(self, func: Callable[[_TCoord, _TCoord], _TCoord]) -> float:
        return NotImplemented

    # def average(self, func: Callable[[_TCoord, _TCoord], _TCoord]) -> float:
    #     return self.integral(func)/self.length

    @cached_property
    def bbox(self) -> np.ndarray:
        """[[xmin,xmax],[ymin,ymax],...]"""
        return np.asarray([[d.min(), d.max()] for d in self.xyz])

    @cached_property
    def center(self):
        return (self.bbox[:, 0]+self.bbox[:, 1])*0.5

    @cached_property
    def is_closed(self):
        if self.rank == 0:
            return True
        else:
            return np.allclose(self.xyz[:, 0], self.xyz[:, -1])

    def trim(self):
        return NotImplemented

    def remesh(self, mesh_type=None, /, **kwargs):
        return NotImplemented

    def encloses_point(self, *x: float, tolerance=None) -> bool:
        x = np.asarray(x)
        bbox = self.bbox
        if np.any(x < bbox[:, 0]) or np.any(x > bbox[:, 1]):
            return False
        elif tolerance is None:
            return True
        else:
            return NotImplemented

    def derivative(self,  *args, **kwargs):
        return NotImplemented

    def pullback(self, func,   *args, **kwargs):
        r"""
            ..math:: f:N\rightarrow M\\\Phi^{*}f:\mathbb{R}\rightarrow M\\\left(\Phi^{*}f\right)\left(u\right)&\equiv f\left(\Phi\left(u\right)\right)=f\left(r\left(u\right),z\left(u\right)\right)
        """
        # if len(args) == 0:
        #     args = self._mesh

        # return Function(args, func(*self.xyz(*args,   **kwargs)), is_period=self.is_closed)
        return NotImplemented
