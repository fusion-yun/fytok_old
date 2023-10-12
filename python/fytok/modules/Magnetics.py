
from .._ontology import magnetics
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Point import Point


class Magnetics(magnetics._T_magnetics):
    """Magnetic diagnostics for equilibrium identification and plasma shape control.
    """

    def __geometry__(self, view_point="RZ", **kwargs) -> GeoObject:
        geo = {}
        styles = {}
        match view_point.lower():
            case "rz":
                geo["b_field_tor_probe"] = [Point(p.position[0].r,  p.position[0].z, name=p.name)
                                            for p in self.b_field_tor_probe]
                geo["flux_loop"] = [Point(p.position[0].r,  p.position[0].z, name=p.name) for p in self.flux_loop]

        return geo, styles
