from __future__ import annotations


from spdm.geometry.GeoObject import GeoObject

from fytok._imas.lastest.pellets import _T_pellets


class Pellets(_T_pellets):
    def __geometry__(self, view="RZ", **kwargs) -> GeoObject:

        return {}
