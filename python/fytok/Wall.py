import collections
from copy import copy

import numpy as np

import matplotlib.pyplot as plt

RZ = collections.namedtuple("RZ", "r z ")


class Wall:
    def __init__(self, *args, **kwargs):
        self._limiter = RZ([], [])
        self._vessel_inner = RZ([], [])
        self._vessel_outer = RZ([], [])

    @property
    def limiter(self):
        return self._limiter

    @limiter.setter
    def limiter(self, rz):
        self._limiter = RZ(*rz)

    @property
    def vessel(self):
        return {"inner": self._vessel_inner, "outer": self._vessel_outer}

    @vessel.setter
    def vessel(self, desc):
        self._vessel_inner = RZ(*desc.get("inner", [[], []]))
        self._vessel_outer = RZ(*desc.get("outer", [[], []]))

    def plot(self, axis=None, **kwargs):

        if axis is None:
            axis = plt.gca()

        vessel_inner_points = np.array([self._vessel_inner.r,  self._vessel_inner.z]).transpose([1, 0])

        vessel_outer_points = np.array([self._vessel_outer.r, self._vessel_outer.z]).transpose([1, 0])

        limiter_points = np.array([self.limiter.r, self.limiter.z]).transpose([1, 0])

        axis.add_patch(plt.Polygon(limiter_points, fill=False, closed=True), **kwargs)
        axis.add_patch(plt.Polygon(vessel_outer_points, fill=False, closed=True), **kwargs)
        axis.add_patch(plt.Polygon(vessel_inner_points, fill=False, closed=True), **kwargs)
        return axis
