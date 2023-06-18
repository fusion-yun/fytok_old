import collections
import typing

import matplotlib.pyplot as plt
import numpy as np
from fytok._imas.lastest.wall import _T_wall, _T_wall_2d
from spdm.data.List import AoS, List
from spdm.data.sp_property import SpDict, sp_property
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Polygon import Polygon
from spdm.utils.logger import logger


class Wall2D(_T_wall_2d):

    def __geometry__(self) -> GeoObject | typing.List[GeoObject]:

        return {"limiter": Polygon(self.limiter.unit[0].outline.r,
                                   self.limiter.unit[0].outline.z),

                "vessel_inner": Polygon(self.vessel.unit[0].annular.outline_inner.r,
                                        self.vessel.unit[0].annular.outline_inner.z),

                "vessel_outer": Polygon(self.vessel.unit[0].annular.outline_outer.r,
                                        self.vessel.unit[0].annular.outline_outer.z)}


class Wall(_T_wall):

    description_2d: AoS[Wall2D] = sp_property()

    def plot(self, axis=None, *args, **kwargs):

        if axis is None:
            axis = plt.gca()

        desc2d = self.description_2d[0]

        # outline = desc2d.vessel.unit[0].annular.outline_inner

        vessel_inner_points = np.array([desc2d.vessel.unit[0].annular.outline_inner.r,
                                        desc2d.vessel.unit[0].annular.outline_inner.z]).transpose([1, 0])

        vessel_outer_points = np.array([desc2d.vessel.unit[0].annular.outline_outer.r,
                                        desc2d.vessel.unit[0].annular.outline_outer.z]).transpose([1, 0])

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

    def __geometry__(self) -> GeoObject | typing.List[GeoObject]:

        return {"description_2d": self.description_2d[0].__geometry__()}
