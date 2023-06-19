import typing

from fytok._imas.lastest.wall import _T_wall

from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Polyline import Polyline
from spdm.utils.logger import logger


class Wall(_T_wall):
    @property
    def __geometry__(self) -> GeoObject | typing.List[GeoObject]:
        desc = self.description_2d[0]  # 0 for equilibrium codes
        return {
            "limiter": Polyline(desc.limiter.unit[0].outline.r,
                                desc.limiter.unit[0].outline.z),

            "vessel_inner": Polyline(desc.vessel.unit[0].annular.outline_inner.r,
                                     desc.vessel.unit[0].annular.outline_inner.z),

            "vessel_outer": Polyline(desc.vessel.unit[0].annular.outline_outer.r,
                                     desc.vessel.unit[0].annular.outline_outer.z)
        }
