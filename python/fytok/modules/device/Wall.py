import collections

import matplotlib.pyplot as plt
import numpy as np
from spdm.data import Dict, Function, Link, List, Node, Signal, sp_property
from spdm.logger import logger
from sympy import Point, Polygon

from ...IDS import IDS
from ..common.Misc import RZTuple


class WallGlobalQuantities(Dict):
    pass


class WallLimiter(Dict):

    class Unit(Dict):
        outline: RZTuple = sp_property()

    unit:  List[Unit] = sp_property()


class WallVessel(Dict):

    class Annular(Dict):

        outline_outer: RZTuple = sp_property(default={})

        outline_inner: RZTuple = sp_property(default={})

    annular: Annular = sp_property()


class WallDescription2D(Dict):

    Limiter = WallLimiter
    Vessel = WallVessel

    limiter: WallLimiter = sp_property()

    vessel: WallVessel = sp_property()

    def limiter_polygon(self):
        limiter_points = np.array([self.limiter.unit[0].outline.r,
                                   self.limiter.unit[0].outline.z]).transpose([1, 0])

        return Polygon(*map(Point, limiter_points))

    def vessel_polygon(self):
        vessel_inner_points = np.array([self.vessel.annular.outline_inner.r,
                                        self.vessel.annular.outline_inner.z]).transpose([1, 0])

        vessel_outer_points = np.array([self.vessel.annular.outline_outer.r,
                                        self.vessel.annular.outline_outer.z]).transpose([1, 0])

        return Polygon(*map(Point, vessel_inner_points)), Polygon(*map(Point, vessel_outer_points))

    # def in_limiter(self, *x):
    #     return self.limiter_polygon().encloses(Point(*x))

    # def in_vessel(self, *x):
    #     return self.vessel_polygon().encloses(Point(*x))

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


class WallDescriptionGGD(Dict):
    pass


class Wall(IDS):
    """Wall

    """
    _IDS = "wall"
    Description2D = WallDescription2D
    DescriptionGGD = WallDescriptionGGD
    GlobalQuantities = WallGlobalQuantities

    global_quantities: GlobalQuantities = sp_property()

    description_2d: List[Description2D] = sp_property()

    description_ggd: List[DescriptionGGD] = sp_property()

    def plot(self, axis=None, *args, **kwargs):

        if axis is None:
            axis = plt.gca()

        desc2d = self.description_2d[0]

        vessel_inner_points = np.array([desc2d.vessel.annular.outline_inner.r,
                                        desc2d.vessel.annular.outline_inner.z]).transpose([1, 0])

        vessel_outer_points = np.array([desc2d.vessel.annular.outline_outer.r,
                                        desc2d.vessel.annular.outline_outer.z]).transpose([1, 0])

        limiter_points = np.array([desc2d.limiter.unit[0].outline.r,
                                   desc2d.limiter.unit[0].outline.z]).transpose([1, 0])

        axis.add_patch(plt.Polygon(limiter_points, **
                                   collections.ChainMap(kwargs.get("limiter", {}), {"fill": False, "closed": True})))

        axis.add_patch(plt.Polygon(vessel_outer_points, **collections.ChainMap(kwargs.get("vessel_outer", {}),
                                                                               kwargs.get("vessel", {}),
                                                                               {"fill": False, "closed": True})))

        axis.add_patch(plt.Polygon(vessel_inner_points, **collections.ChainMap(kwargs.get("vessel_inner", {}),
                                                                               kwargs.get("vessel", {}),
                                                                               {"fill": False, "closed": True})))

        return axis
