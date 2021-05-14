import collections
from copy import copy
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.AttributeTree import as_attribute_tree
from spdm.data.Node import Dict, List
from spdm.util.logger import logger
from sympy import Point, Polygon
from ..common.IDS import IDS


class WallDescription2D(Dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def limiter_polygon(self):
        limiter_points = np.array([self.limiter.unit.outline.r,
                                   self.limiter.unit.outline.z]).transpose([1, 0])

        return Polygon(*map(Point, limiter_points))

    @cached_property
    def vessel_polygon(self):
        vessel_inner_points = np.array([self.vessel.annular.outline_inner.r,
                                        self.vessel.annular.outline_inner.z]).transpose([1, 0])

        vessel_outer_points = np.array([self.vessel.annular.outline_outer.r,
                                        self.vessel.annular.outline_outer.z]).transpose([1, 0])

        return Polygon(*map(Point, vessel_inner_points)), Polygon(*map(Point, vessel_outer_points))

    def in_limiter(self, *x):
        return self.limiter_polygon.encloses(Point(*x))

    def in_vessel(self, *x):
        return self.vessel_polygon.encloses(Point(*x))

        # if isinstance(data, LazyProxy):
        #     limiter = data.description_2d.limiter.unit.outline()
        #     vessel = data.description_2d.vessel.annular()
        # else:
        #     limiter = None
        #     vessel = None

        # if limiter is None:
        #     pass
        # elif isinstance(limiter, list):
        #     self.limiter.outline.r = limiter[0]
        #     self.limiter.outline.z = limiter[1]
        # elif isinstance(limiter, collections.abc.Mapping):
        #     self.limiter.outline.r = limiter["r"]
        #     self.limiter.outline.z = limiter["z"]
        # elif isinstance(limiter, LazyProxy):
        #     self.limiter.outline.r = limiter.r
        #     self.limiter.outline.z = limiter.z
        # else:
        #     raise TypeError(f"Unknown type {type(limiter)}")

        # if vessel is None:
        #     pass
        # elif isinstance(vessel, list):
        #     self.vessel.annular.outline_inner.r = vessel[0][0]
        #     self.vessel.annular.outline_inner.z = vessel[0][1]
        #     self.vessel.annular.outline_outer.r = vessel[1][0]
        #     self.vessel.annular.outline_outer.z = vessel[1][1]
        # elif isinstance(limiter, collections.abc.Mapping):
        #     self.vessel.annular.outline_inner.r = vessel["outline_inner"]["r"]
        #     self.vessel.annular.outline_inner.z = vessel["outline_inner"]["z"]
        #     self.vessel.annular.outline_outer.r = vessel["outline_outer"]["r"]
        #     self.vessel.annular.outline_outer.z = vessel["outline_outer"]["z"]
        # elif isinstance(limiter, LazyProxy):
        #     self.vessel.annular.outline_inner.r = vessel.outline_inner.r
        #     self.vessel.annular.outline_inner.z = vessel.outline_inner.z
        #     self.vessel.annular.outline_outer.r = vessel.outline_outer.r
        #     self.vessel.annular.outline_outer.z = vessel.outline_outer.z
        # else:
        #     raise TypeError(f"Unknown type {type(vessel)}")


class Wall(IDS):
    """Wall

    """
    _IDS = "wall"
    _homogeneous_time= 2

    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def description_2d(self) -> WallDescription2D:
        return WallDescription2D(self["description_2d"], parent=self)

    def plot(self, axis=None, *args, **kwargs):

        if axis is None:
            axis = plt.gca()

        desc2d = self.description_2d

        vessel_inner_points = np.array([desc2d['vessel.annular.outline_inner.r'],
                                        desc2d["vessel.annular.outline_inner.z"]]).transpose([1, 0])

        vessel_outer_points = np.array([desc2d["vessel.annular.outline_outer.r"],
                                        desc2d["vessel.annular.outline_outer.z"]]).transpose([1, 0])

        limiter_points = np.array([desc2d["limiter.unit.outline.r"],
                                   desc2d["limiter.unit.outline.z"]]).transpose([1, 0])

        axis.add_patch(plt.Polygon(limiter_points,  **
                                   collections.ChainMap(kwargs.get("limiter", {}), {"fill": False, "closed": True})))

        axis.add_patch(plt.Polygon(vessel_outer_points, **collections.ChainMap(kwargs.get("vessel_outer", {}),
                                                                               kwargs.get("vessel", {}),
                                                                               {"fill": False, "closed": True})))

        axis.add_patch(plt.Polygon(vessel_inner_points, **collections.ChainMap(kwargs.get("vessel_inner", {}),
                                                                               kwargs.get("vessel", {}),
                                                                               {"fill": False, "closed": True})))

        return axis
