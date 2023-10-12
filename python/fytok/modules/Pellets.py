from __future__ import annotations


from spdm.geometry.GeoObject import GeoObject

from .._ontology import pellets


class Pellets(pellets._T_pellets):
    def __geometry__(self, view="RZ", **kwargs) -> GeoObject:

        return {}
