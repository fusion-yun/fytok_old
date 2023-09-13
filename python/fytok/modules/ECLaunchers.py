from __future__ import annotations


from spdm.geometry.GeoObject import GeoObject

from fytok._imas.lastest.ec_launchers import _T_ec_launchers


class ECLaunchers(_T_ec_launchers):
    def __geometry__(self, view="RZ", **kwargs) -> GeoObject:

        return {}
