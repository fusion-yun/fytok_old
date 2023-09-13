from __future__ import annotations

import numpy as np
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Polygon import Polygon
from spdm.utils.constants import TWOPI
from spdm.utils.logger import logger

from fytok._imas.lastest.tf import _T_tf


class TF(_T_tf):

    def __geometry__(self, view="RZ", **kwargs) -> GeoObject:
        geo = {}
        styles = {}
        r0 = self.r0

        if view == "RZ":
            coils = [
                Polygon(conductor.elements.start_points.r,
                        conductor.elements.start_points.z,
                        name=self.coil[0].name)
                for conductor in self.coil[0].conductor]

        else:
            if self.is_periodic == 0:
                coils_n = self.coils_n
                d_phi = TWOPI/self.coils_n
                logger.debug(d_phi)
                cross_section = self.coil[0].conductor[0].cross_section
                r = cross_section.delta_r
                phi = cross_section.delta_phi
                name = self.coil[0].name
                coils = [
                    Polygon((r0+r)*np.cos(phi+d_phi*i),
                            (r0+r)*np.sin(phi+d_phi*i),
                            name=name+f"{i}") for i in range(coils_n)]

            else:
                coils = [
                    Polygon(
                        (r0 + coil.conductor[0].cross_section.delta_r) *
                        np.cos(coil.conductor[0].cross_section.delta_phi),
                        (r0 + coil.conductor[0].cross_section.delta_r) *
                        np.sin(coil.conductor[0].cross_section.delta_phi),
                        name=coil.name) for coil in self.coil]

            geo["coils"] = coils

        return geo, styles
