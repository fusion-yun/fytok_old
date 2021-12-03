import collections

import matplotlib.pyplot as plt
import numpy as np
from spdm.common.logger import logger
from spdm.data.Dict import Dict
from spdm.data.Entry import Entry
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property
from sympy import Point, Polygon

from ..common.IDS import IDS
from ..common.Misc import RZTuple


class WallGlobalQuantities(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class WallLimiter(Dict):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    class Unit(Dict):
        def __init__(self,  *args, **kwargs):
            super().__init__(*args, **kwargs)

        @sp_property
        def outline(self) -> RZTuple:
            return self.get("outline", {})

    @sp_property
    def unit(self) -> List[Unit]:
        return self.get("unit")


class WallVessel(Dict):
    def __init__(self,   *args, **kwargs):
        super().__init__(*args, **kwargs)

    class Annular(Dict):
        def __init__(self,  *args, **kwargs):
            super().__init__(*args, **kwargs)

        @sp_property
        def outline_outer(self) -> RZTuple:
            return self.get("outline_outer", {})

        @sp_property
        def outline_inner(self) -> RZTuple:
            return self.get("outline_inner", {})

    @sp_property
    def annular(self) -> Annular:
        return self.get("annular")


class WallDescription2D(Dict):

    Limiter = WallLimiter
    Vessel = WallVessel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def limiter(self) -> Limiter:
        return self.get("limiter")

    @sp_property
    def vessel(self) -> Vessel:
        return self.get("vessel")

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Wall(IDS):
    """Wall

    """
    _IDS = "wall"
    Description2D = WallDescription2D
    DescriptionGGD = WallDescriptionGGD
    GlobalQuantities = WallGlobalQuantities

    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def global_quantities(self) -> GlobalQuantities:
        return self.get("global_quantities")

    @sp_property
    def description_2d(self) -> List[Description2D]:
        return self.get("description_2d")

    @sp_property
    def description_ggd(self) -> List[DescriptionGGD]:
        return self.get("description_ggd")

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
