from __future__ import annotations

from spdm.geometry.GeoObject import GeoObject

from .._imas.lastest.lh_antennas import _T_lh_antennas


class LHAntennas(_T_lh_antennas):
    def __geometry__(self, view_point="RZ", **kwargs) -> GeoObject:

        geo = {}
        styles = {}
        match view_point.lower():
            case "top":
                geo["antenna"] = [antenna.name for antenna in self.antenna]
                styles["antenna"] = {"$matplotlib": {"color": 'blue'}, "text": True}

        return geo, styles
