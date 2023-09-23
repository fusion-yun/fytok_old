import typing

from .._imas.lastest.wall import _T_wall

from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Polyline import Polyline
from spdm.geometry.Circle import Circle
from ..utils.logger import logger


class Wall(_T_wall):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __geometry__(self, view="RZ", **kwargs) -> GeoObject | typing.List[GeoObject]:

        geo = {}
        styles = {}

        desc = self.description_2d[0]  # 0 for equilibrium codes
        try:
            match view.lower():
                case "top":
                    vessel_r = desc.vessel.unit[0].annular.outline_outer.r
                    vessel_z = desc.vessel.unit[0].annular.outline_outer.z
                    geo["vessel_outer"] = [Circle(0.0, 0.0, vessel_r.min()), Circle(0.0, 0.0, vessel_r.max())]

                case "rz":
                    geo["limiter"] = Polyline(desc.limiter.unit[0].outline.r,
                                              desc.limiter.unit[0].outline.z)

                    vessel = desc.vessel.unit[0].annular

                    geo["vessel_inner"] = Polyline(vessel.outline_inner.r,
                                                   vessel.outline_inner.z)

                    geo["vessel_outer"] = Polyline(vessel.outline_outer.r,
                                                   vessel.outline_outer.z)
        except Exception:
            logger.debug(f"Geometry of {self.__class__.__name__} is incomplete!")
            pass

        styles = {  #
            "limiter": {"$matplotlib": {"edgecolor": "green"}},
            "vessel_inner": {"$matplotlib": {"edgecolor": "blue"}},
            "vessel_outer": {"$matplotlib": {"edgecolor": "blue"}}
        }
        return geo, styles
