import typing
from fytok._imas.lastest.pf_active import _T_pf_active

from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Polygon import Rectangle
from spdm.utils.logger import logger


class PFActive(_T_pf_active):
    @property
    def __geometry__(self) -> GeoObject | typing.Container[GeoObject]:
        geo_coils = []
        for coil in self.coil:
            rect = coil.element[0].geometry.rectangle
            geo_coils.append(Rectangle(rect.r - rect.width / 2.0,  rect.z -
                             rect.height / 2.0,   rect.width,  rect.height,
                             name=coil.name))

        return {"coil": geo_coils}, {"coil": {"$matplotlib": {"color": 'black'}, "title": True}},
