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

from .CoreProfiles import CoreProfiles
from .PFActive import PFActive
from .Wall import Wall


class EquilibriumTimeSlice(_T_equilibrium_time_slice):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._R0 = self.get("../vacuum_toroidal_field/r0")
        self._B0 = self.get("../vacuum_toroidal_field/b0")(self.time)
       

    def update(self, *args,   **kwargs) -> EquilibriumTimeSlice:
        logger.debug(f"Update Equlibrium at time={self.time}")
        super().update(*args, **kwargs)
        return self

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
            logger.error(f"Can not get o-point/x-point! {error}")

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

# def plot(self, axis=None, *args,
    #          scalar_field={},
    #          vector_field={},
    #          boundary=True,
    #          separatrix=True,
    #          contours=16,
    #          oxpoints=True,
    #          **kwargs):
    #     """
    #         plot o-point,x-point,lcfs,separatrix and contour of psi
    #     """

    #     import matplotlib.pyplot as plt

    #     if axis is None:
    #         axis = plt.gca()

    #     if oxpoints:
    #         try:
    #             o_points, x_points = self.coordinate_system.critical_points
    #             for idx, o in enumerate(o_points):
    #                 if idx == 0:
    #                     axis.plot(o.r, o.z, 'g+', linewidth=0.5, label="Magnetic axis")
    #                 else:
    #                     axis.plot(o.r, o.z, 'g+',)
    #                     axis.text(o.r, o.z, idx,
    #                               horizontalalignment='center',
    #                               verticalalignment='center')
    #             for idx, x in enumerate(x_points):
    #                 axis.plot(x.r, x.z, 'rx')
    #                 axis.text(x.r, x.z, idx,
    #                           horizontalalignment='center',
    #                           verticalalignment='center')
    #         except Exception as error:
    #             logger.error(f"Can not find o-point/x-point! {error}")

    #     if boundary:
    #         try:

    #             boundary_points = np.vstack([self.boundary.outline.r.__array__(),
    #                                         self.boundary.outline.z.__array__()]).T

    #             axis.add_patch(plt.Polygon(boundary_points, color='r', linestyle='solid',
    #                                        linewidth=0.5, fill=False, closed=True))

    #             # axis.plot([], [], 'g-', label="Boundary")
    #         except Exception as error:
    #             logger.error(f"Plot boundary failed! {error}")
    #             # raise RuntimeError(f"Plot boundary failed!") from error

    #     if separatrix:
    #         try:
    #             separatrix_outline = np.vstack([self.boundary_separatrix.outline.r.__array__(),
    #                                             self.boundary_separatrix.outline.z.__array__()]).T

    #             axis.add_patch(plt.Polygon(separatrix_outline, color='r', linestyle='dashed',
    #                                        linewidth=0.5, fill=False, closed=False))
    #             axis.plot([], [], 'r--', label="Separatrix")

    #         except Exception as error:
    #             logger.error(f"Plot separatrix failed! {error}")

    #         # p = self.boundary_separatrix.geometric_axis
    #         # axis.plot(p.r, p.z, 'rx')
    #         # axis.text(p.r, p.z, 'x',
    #         #           horizontalalignment='center',
    #         #           verticalalignment='center')
    #         # axis.plot([], [], 'rx', label="X-Point")

    #         # for idx, p in self.boundary_secondary_separatrix.x_point:
    #         #     axis.plot(p.r, p.z, 'rx')
    #         #     axis.text(p.r, p.z, f'x{idx}',
    #         #               horizontalalignment='center',
    #         #               verticalalignment='center')

    #         # for idx, p in self.boundary_secondary_separatrix.strike_point:
    #         #     axis.plot(p.r, p.z, 'rx')
    #         #     axis.text(p.r, p.z, f's{idx}',
    #         #               horizontalalignment='center',
    #         #               verticalalignment='center')

    #     if contours:
    #         if contours is True:
    #             contours = 16
    #         profiles_2d = self.profiles_2d[0]

    #         try:
    #             axis.contour(profiles_2d.r.__array__(),
    #                          profiles_2d.z.__array__(),
    #                          profiles_2d.psi.__array__(), linewidths=0.5, levels=contours)
    #         except Exception as error:
    #             logger.error(f"Plot contour of psi failed! {error}")

    #     for s, opts in scalar_field.items():
    #         if s == "psirz":
    #             self.coordinate_system._psirz.plot(axis, **opts)
    #         else:
    #             sf = getattr(profiles_2d, s, None)

    #             if isinstance(sf,  Function):
    #                 sf = sf(profiles_2d.r, profiles_2d.z)

    #             if isinstance(sf, np.ndarray):
    #                 axis.contour(profiles_2d.r, profiles_2d.z, sf, linewidths=0.2, **opts)
    #             else:
    #                 logger.error(f"Can not find field {sf} {type(sf)}!")

    #     for u, v, opts in vector_field.items():
    #         uf = profiles_2d[u]
    #         vf = profiles_2d[v]
    #         axis.streamplot(profiles_2d.grid.dim1, profiles_2d.grid.dim2, vf, uf, **opts)

    #     return axis


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    TimeSlice = EquilibriumTimeSlice
    time_slice: TimeSeriesAoS[TimeSlice] = sp_property(coordinate1="time", type="dynamic")

    _plugin_registry = {}

    def update(self, *args,
               core_profile_1d: CoreProfiles.Profiles1d = None,
               pf_active: PFActive = None,
               wall: Wall = None,  **kwargs) -> TimeSlice:
        """ update the last time slice """
        return super().update(*args, **kwargs)

    def advance(self, *args, time: float = 0.0,
                core_profile_1d: CoreProfiles.Profiles1d = None,
                pf_active: PFActive = None,
                wall: Wall = None, **kwargs) -> Equilibrium.TimeSlice:
        super().advance(time=time)
        return super().update(*args, **kwargs)

    @ property
    def __geometry__(self) -> GeoObject | typing.Container[GeoObject]:
        return self.time_slice.current.__geometry__

    # def plot(self, axis,  *args, time_slice=-1,  **kwargs):
    #     if len(self.time_slice) == 0 or time_slice >= len(self.time_slice):
    #         logger.error(f"Time slice {time_slice} is out range {len(self.time_slice)} !")
    #         return axis
    #     else:
    #         return self.time_slice[time_slice].plot(axis, *args, **kwargs)
