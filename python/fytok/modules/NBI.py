from __future__ import annotations

import typing

from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Line import Line
from spdm.geometry.Polygon import Rectangle

from fytok._imas.lastest.nbi import _T_nbi, _T_nbi_unit


def draw_nbi_unit(unit: _T_nbi_unit, name: str):
    geo = None
    if unit.source.geometry_type == 3:
        geo = [
            Line(),
            Rectangle(name=unit.name)]

    else:
        pass

    return geo


class NBI(_T_nbi):
    def __geometry__(self, view="RZ", **kwargs) -> GeoObject:
        geo = {}
        styles = {}

        if view == "RZ":
            pass
        else:
            geo["unit"] = [draw_nbi_unit(unit) for unit in self.unit]

        return geo, styles
