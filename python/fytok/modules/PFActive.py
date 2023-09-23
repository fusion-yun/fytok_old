import typing

from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Polygon import Rectangle
from ..utils.logger import logger

from .._imas.lastest.pf_active import _T_pf_active


class PFActive(_T_pf_active):
    def __geometry__(self, view="RZ", **kwargs) -> GeoObject:
        geo = {}
        styles = {}

        if view != "RZ":
            pass
        else:
            geo_coils = []
            for coil in self.coil:
                rect = coil.element[0].geometry.rectangle
                geo_coils.append(Rectangle(rect.r - rect.width / 2.0,  rect.z -
                                           rect.height / 2.0,   rect.width,  rect.height,
                                           name=coil.name))

            geo["coil"] = geo_coils
            styles["coil"] = {"$matplotlib": {"color": 'black'}, "text": True}

        return geo, styles
