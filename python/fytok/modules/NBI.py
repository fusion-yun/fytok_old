from __future__ import annotations

import typing

from spdm.geometry.GeoObject import GeoObject
from fytok._imas.lastest.nbi import _T_nbi


class NBI(_T_nbi):
    def __geometry__(self, view="RZ", **kwargs) -> GeoObject:

        return {}
