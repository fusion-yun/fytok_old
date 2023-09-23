from __future__ import annotations

import numpy as np
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Polygon import Polygon
from spdm.utils.constants import TWOPI
from ..utils.logger import logger
from spdm.utils.typing import _not_found_

from .._imas.lastest.tf import _T_tf


class TF(_T_tf):

    def __geometry__(self, view="RZ", **kwargs) -> GeoObject:
        geo = {}
        styles = {}
        r0 = self.r0
        try:
            match view.lower():
                case "rz":
                    conductor = self.coil[0].conductor[0]
                    if conductor.elements.start_points.r is not _not_found_:
                        geo["coils"] = Polygon(conductor.elements.start_points.r,
                                               conductor.elements.start_points.z,
                                               name=self.coil[0].name)

                case "top":
                    if self.is_periodic == 0:
                        coils_n = self.coils_n
                        d_phi = TWOPI/self.coils_n

                        cross_section = self.coil[0].conductor[0].cross_section
                        r = cross_section.delta_r
                        phi = cross_section.delta_phi
                        name = self.coil[0].name
                        geo["coils"] = [
                            Polygon((r0+r)*np.cos(phi+d_phi*i),
                                    (r0+r)*np.sin(phi+d_phi*i),
                                    name=name+f"{i}") for i in range(coils_n)]

                    else:
                        geo["coils"] = [
                            Polygon(
                                (r0 + coil.conductor[0].cross_section.delta_r) *
                                np.cos(coil.conductor[0].cross_section.delta_phi),
                                (r0 + coil.conductor[0].cross_section.delta_r) *
                                np.sin(coil.conductor[0].cross_section.delta_phi),
                                name=coil.name) for coil in self.coil]
        except Exception:
            logger.warning(f"Geometry of {self.__class__.__name__} is incomplete!")
        return geo, styles
