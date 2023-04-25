
from functools import cached_property
from operator import is_

from spdm.numlib import interpolate, np

from ..data.Function import Function
from spdm.utils.logger import logger
from .Curve import Curve


class BSplineCurve(Curve):
    def __init__(self, u, p, *args, is_closed=None, cycle=None, **kwargs) -> None:
        # if len(args) != 2:
        #     raise ValueError(f"Illegal input! len(args)={len(args)}")
        super().__init__(*args, is_closed=is_closed is not None, cycle=cycle)

        self._u = u if u is not None else np.linspace(0, 1.0, len(p[0]))
        self._spl = interpolate.make_interp_spline(self._u, np.c_[tuple(p)], **kwargs)
        if self.is_closed:
            self._spl.extrapolate = 'periodic'

    def inside(self, *x):
        return NotImplemented

    @property
    def points(self):
        return self.map(self._u)

    def point(self, u,  *args, **kwargs):
        return self._spl(u,  *args, **kwargs).T

    def map(self, u, *args, **kwargs):
        r"""
            ..math:: \Phi:\mathbb{R}\rightarrow N
        """
        return self._spl(u, *args, **kwargs).T

    def __call__(self, *args, **kwargs):
        return self.map(*args, **kwargs)

    @cached_property
    def _derivative(self):
        return self._spl.derivative()

    def derivative(self,  *args, **kwargs):
        if len(args) == 0:
            return self._derivative(self._u, **kwargs).T
        else:
            return self._derivative(*args, **kwargs).T

    def pullback(self, func, *args, form=0, **kwargs):
        if len(args) > 0:
            return func(*self.map(*args, **kwargs))

        if form == 0:
            fun = Function(self._u, func(*self.map(self._u)), is_period=self.is_closed)
        elif form == 1:
            fun = Function(self._u, func(*self.map(self._u))*self.dl(self._u), is_period=self.is_closed)
        else:
            raise ValueError()

        return fun
