import collections
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from spdm.util.LazyProxy import LazyProxy
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger


class Wall(AttributeTree):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args)+len(kwargs) > 0:
            self.load(*args, **kwargs)

    def load(self, ids=None, *args, limiter=None, vessel=None, **kwargs):
        if isinstance(ids, LazyProxy):
            if limiter is None:
                limiter = ids.description_2d.limiter.unit.outline

            if vessel is None:
                vessel = ids.description_2d.vessel.annular

        if limiter is None:
            pass
        elif isinstance(limiter, list):
            self.limiter.outline.r = limiter[0]
            self.limiter.outline.z = limiter[1]
        elif isinstance(limiter, collections.abc.Mapping):
            self.limiter.outline.r = limiter["r"]
            self.limiter.outline.z = limiter["z"]
        elif isinstance(limiter, LazyProxy):
            self.limiter.outline.r = limiter.r()
            self.limiter.outline.z = limiter.z()
        else:
            raise TypeError(f"Unknown type {type(limiter)}")

        if vessel is None:
            pass
        elif isinstance(vessel, list):
            self.vessel.annular.outline_inner.r = vessel[0][0]
            self.vessel.annular.outline_inner.z = vessel[0][1]
            self.vessel.annular.outline_outer.r = vessel[1][0]
            self.vessel.annular.outline_outer.z = vessel[1][1]
        elif isinstance(limiter, collections.abc.Mapping):
            self.vessel.annular.outline_inner.r = vessel["outline_inner"]["r"]
            self.vessel.annular.outline_inner.z = vessel["outline_inner"]["z"]
            self.vessel.annular.outline_outer.r = vessel["outline_outer"]["r"]
            self.vessel.annular.outline_outer.z = vessel["outline_outer"]["z"]
        elif isinstance(limiter, LazyProxy):
            self.vessel.annular.outline_inner.r = vessel.outline_inner.r()
            self.vessel.annular.outline_inner.z = vessel.outline_inner.z()
            self.vessel.annular.outline_outer.r = vessel.outline_outer.r()
            self.vessel.annular.outline_outer.z = vessel.outline_outer.z()
        else:
            raise TypeError(f"Unknown type {type(vessel)}")
        
        return self.entry

 
    def plot(self, axis=None, *args, **kwargs):

        if axis is None:
            axis = plt.gca()

        vessel_inner_points = np.array([self.vessel.annular.outline_inner.r,
                                        self.vessel.annular.outline_inner.z]).transpose([1, 0])

        vessel_outer_points = np.array([self.vessel.annular.outline_outer.r,
                                        self.vessel.annular.outline_outer.z]).transpose([1, 0])

        limiter_points = np.array([self.limiter.outline.r, self.limiter.outline.z]).transpose([1, 0])

        axis.add_patch(plt.Polygon(limiter_points, fill=False, closed=True))
        axis.add_patch(plt.Polygon(vessel_outer_points, fill=False, closed=True))
        axis.add_patch(plt.Polygon(vessel_inner_points, fill=False, closed=True))

        return axis
