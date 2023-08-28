from __future__ import annotations

import typing

from fytok._imas.lastest.equilibrium import (
    _T_equilibrium, _T_equilibrium_boundary,
    _T_equilibrium_boundary_separatrix, _T_equilibrium_constraints,
    _T_equilibrium_coordinate_system, _T_equilibrium_global_quantities,
    _T_equilibrium_profiles_1d, _T_equilibrium_profiles_2d,
    _T_equilibrium_time_slice)
from spdm.data.sp_property import sp_property
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.geometry.Curve import Curve
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Point import Point
from spdm.utils.logger import logger


class EquilibriumTimeSlice(_T_equilibrium_time_slice):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._R0 = self.get("../vacuum_toroidal_field/r0")
        self._B0 = self.get("../vacuum_toroidal_field/b0")(self.time)

    CoordinateSystem = _T_equilibrium_coordinate_system
    Profiles1d = _T_equilibrium_profiles_1d
    Profiles2d = _T_equilibrium_profiles_2d
    GlobalQuantities = _T_equilibrium_global_quantities
    Boundary = _T_equilibrium_boundary
    BoundarySeparatrix = _T_equilibrium_boundary_separatrix
    Constraints = _T_equilibrium_constraints

    @property
    def __geometry__(self) -> GeoObject | typing.Container[GeoObject]:

        geo = {}

        try:
            o_points, x_points = self.coordinate_system.critical_points

            geo["o_points"] = [Point(p.r, p.z, name=f"{idx}") for idx, p in enumerate(o_points)]
            geo["x_points"] = [Point(p.r, p.z, name=f"{idx}") for idx, p in enumerate(x_points)]

        except Exception as error:
            raise RuntimeError(f"Can not get o-point/x-point! {error}") from error

        try:
            geo["boundary"] = Curve(self.boundary.outline.r.__array__(),
                                    self.boundary.outline.z.__array__())
        except Exception as error:
            logger.error(f"Can not get boundary! {error}")

        try:
            geo["boundary_separatrix"] = Curve(self.boundary_separatrix.outline.r.__array__(),
                                               self.boundary_separatrix.outline.z.__array__())
        except Exception as error:
            logger.error(f"Can not get boundary_separatrix! {error}")

        geo["psi"] = self.profiles_2d[0].psi

        styles = {
            "o_points":  {"$matplotlib": {"c": 'red', 'marker': '.'}},
            "x_points":  {"$matplotlib": {"c": 'blue', 'marker': 'x'}},
            "boundary":  {"$matplotlib": {"color": 'red', 'linewidth': 0.5}},
            "boundary_separatrix":  {"$matplotlib": {"color": 'red', "linestyle": 'dashed', 'linewidth': 0.25}},
        }
        return geo, styles


class Equilibrium(_T_equilibrium):
    r"""
        Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.

        Reference:
            - O. Sauter and S. Yu Medvedev, "Tokamak coordinate conventions: COCOS", Computer Physics Communications 184, 2 (2013), pp. 293--302.

        COCOS  11
        ```{text}
            Top view
                     ***************
                    *               *
                   *   ***********   *
                  *   *           *   *
                 *   *             *   *
                 *   *             *   *
             Ip  v   *             *   ^  \phi
                 *   *    Z o--->R *   *
                 *   *             *   *
                 *   *             *   *
                 *   *     Bpol    *   *
                  *   *     o     *   *
                   *   ***********   *
                    *               *
                     ***************
                       Bpol x
                    Poloidal view
                ^Z
                |
                |       ************
                |      *            *
                |     *         ^    *
                |     *   \rho /     *
                |     *       /      *
                +-----*------X-------*---->R
                |     *  Ip, \phi   *
                |     *              *
                |      *            *
                |       *****<******
                |       Bpol,\theta
                |
                    Cylindrical coordinate      : $(R,\phi,Z)$
            Poloidal plane coordinate   : $(\rho,\theta,\phi)$
        ```
    """

    _plugin_registry = {}
    
    _plugin_prefix = "fytok/plugins/equilibrium"

    def __init__(self, *args, default_plugin="eq_analyze", **kwargs):
        super().__init__(*args, default_plugin=default_plugin, **kwargs)

    TimeSlice = EquilibriumTimeSlice

    time_slice: TimeSeriesAoS[TimeSlice] = sp_property(coordinate1="time", type="dynamic")

    def refresh(self, *args, **kwargs):
        """ update the last time slice """
        self.time_slice.refresh(*args, **kwargs)
        # self.grids_ggd.refresh(*args, **kwargs)

    def advance(self, *args,  **kwargs):
        self.time_slice.advance(*args, **kwargs)
        # self.grids_ggd.advance(*args, **kwargs)

    @property
    def __geometry__(self) -> GeoObject | typing.Container[GeoObject]:
        return self.time_slice.current.__geometry__

    # def plot(self, axis,  *args, time_slice=-1,  **kwargs):
    #     if len(self.time_slice) == 0 or time_slice >= len(self.time_slice):
    #         logger.error(f"Time slice {time_slice} is out range {len(self.time_slice)} !")
    #         return axis
    #     else:
    #         return self.time_slice[time_slice].plot(axis, *args, **kwargs)
