
from .._imas.lastest.magnetics import _T_magnetics
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Point import Point
from ..utils.logger import logger
import typing


class Magnetics(_T_magnetics):
    """Magnetic diagnostics for equilibrium identification and plasma shape control.
    """
    
    def __geometry__(self, view="RZ", **kwargs) -> GeoObject:
        if view != "RZ":
            return None
        else:
            return {
                "b_field_tor_probe": [Point(p.position[0].r,  p.position[0].z, name=p.name) for p in self.b_field_tor_probe],
                "flux_loop": [Point(p.position[0].r,  p.position[0].z, name=p.name) for p in self.flux_loop]
            }
