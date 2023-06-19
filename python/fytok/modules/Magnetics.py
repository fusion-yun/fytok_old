
from fytok._imas.lastest.magnetics import _T_magnetics
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Point import Point
from spdm.utils.logger import logger
import typing


class Magnetics(_T_magnetics):
    """Magnetic diagnostics for equilibrium identification and plasma shape control.
    """
    @property
    def __geometry__(self) -> GeoObject | typing.Container[GeoObject]:
        return {
            "b_field_tor_probe": [Point(p.position[0].r,  p.position[0].z, name=p.name) for p in self.b_field_tor_probe],
            "flux_loop": [Point(p.position[0].r,  p.position[0].z, name=p.name) for p in self.flux_loop]
        }
