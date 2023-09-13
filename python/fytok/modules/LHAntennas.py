from __future__ import annotations

from spdm.geometry.GeoObject import GeoObject

from fytok._imas.lastest.lh_antennas import _T_lh_antennas


class LHAntennas(_T_lh_antennas):
    def __geometry__(self, view="RZ", **kwargs) -> GeoObject:

        geo = {}
        styles = {}
        if view != "RZ":
            geo["antenna"] = [antenna.name for antenna in self.antenna]
            styles["antenna"] = {"$matplotlib": {"color": 'blue'}, "text": True}

        return geo, styles
