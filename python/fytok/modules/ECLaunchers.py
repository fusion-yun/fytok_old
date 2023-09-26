from __future__ import annotations


from spdm.geometry.GeoObject import GeoObject

from .._imas.lastest.ec_launchers import _T_ec_launchers


class ECLaunchers(_T_ec_launchers):
    def __geometry__(self, view_point="RZ", **kwargs) -> GeoObject:

        geo = {}
        styles = {}
        match view_point.lower():
            case "top":
                geo["beam"] = [beam.name for beam in self.beam]
                styles["beam"] = {"$matplotlib": {"color": 'blue'}, "text": True}

        return geo, styles
