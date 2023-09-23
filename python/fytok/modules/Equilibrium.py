from __future__ import annotations
import numpy as np
import typing

from spdm.data.AoS import AoS
from spdm.data.Function import Function
from spdm.data.sp_property import sp_property
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.geometry.Curve import Curve
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Point import Point
from spdm.utils.tags import _not_found_
from spdm.utils.tree_utils import merge_tree_recursive

from .._imas.lastest.equilibrium import (_T_equilibrium,
                                         _T_equilibrium_boundary,
                                         _T_equilibrium_boundary_separatrix,
                                         _T_equilibrium_constraints,
                                         _T_equilibrium_coordinate_system,
                                         _T_equilibrium_global_quantities,
                                         _T_equilibrium_profiles_1d,
                                         _T_equilibrium_profiles_2d,
                                         _T_equilibrium_time_slice)
from .._imas.lastest.utilities import (_T_b_tor_vacuum_aos3,
                                       _T_equilibrium_profiles_2d_grid)
from ..utils.logger import logger


class EquilibriumCoordinateSystem(_T_equilibrium_coordinate_system):
    pass


class EquilibriumGlobalQuantities(_T_equilibrium_global_quantities):
    pass


class EquilibriumProfiles1D(_T_equilibrium_profiles_1d):

    minor_radius: Function = sp_property(type="dynamic", coordinate1="../psi", units="m")
    """ minor radius """

    triangularity: Function = sp_property(type="dynamic", coordinate1="../psi", units="-")
    """ triangularity """

    squareness: Function = sp_property(type="dynamic", coordinate1="../psi", units="-")
    """ squareness: T. Luce, Plasma Phys. Control. Fusion 55 (2013) 095009 """


class EquilibriumProfiles2D(_T_equilibrium_profiles_2d):
    pass


class EquilibriumTimeSlice(_T_equilibrium_time_slice):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._R0 = self.vacuum_toroidal_field.r0
        self._B0 = self.vacuum_toroidal_field.b0

    CoordinateSystem = EquilibriumCoordinateSystem
    Profiles1D = EquilibriumProfiles1D
    Profiles2D = EquilibriumProfiles2D
    GlobalQuantities = EquilibriumGlobalQuantities
    Boundary = _T_equilibrium_boundary
    BoundarySeparatrix = _T_equilibrium_boundary_separatrix
    Constraints = _T_equilibrium_constraints

    vacuum_toroidal_field: _T_b_tor_vacuum_aos3 = sp_property()

    global_quantities: GlobalQuantities = sp_property()

    profiles_1d: Profiles1D = sp_property()

    profiles_2d: AoS[Profiles2D] = sp_property()

    coordinate_system: CoordinateSystem = sp_property()

    def __geometry__(self, view="RZ", **kwargs) -> GeoObject:
        geo = {}

        try:
            o_points, x_points = self.coordinate_system.critical_points

            geo["o_points"] = [
                Point(p.r, p.z, name=f"{idx}") for idx, p in enumerate(o_points)
            ]
            geo["x_points"] = [
                Point(p.r, p.z, name=f"{idx}") for idx, p in enumerate(x_points)
            ]

            geo["boundary"] = Curve(
                self.boundary.outline.r.__array__(), self.boundary.outline.z.__array__()
            )

            geo["boundary_separatrix"] = Curve(
                self.boundary_separatrix.outline.r.__array__(),
                self.boundary_separatrix.outline.z.__array__(),
            )

        except Exception as error:
            logger.error(f"Can not parser psi ! {error}")
            # raise RuntimeError(f"Can not get o-point/x-point! {error}") from error

        geo["psi"] = self.profiles_2d[0].psi

        styles = {
            "o_points": {"$matplotlib": {"c": "red", "marker": "."}},
            "x_points": {"$matplotlib": {"c": "blue", "marker": "x"}},
            "boundary": {"$matplotlib": {"color": "red", "linewidth": 0.5}},
            "boundary_separatrix": {
                "$matplotlib": {
                    "color": "red",
                    "linestyle": "dashed",
                    "linewidth": 0.25,
                }
            },
        }
        styles = merge_tree_recursive(styles, kwargs)

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
    _plugin_prefix = 'fytok.plugins.equilibrium.'
    _plugin_config = {
        "time_slice": {
            "coordinate_system": {"grid":  {
                "dim1": np.linspace(0, 0.995, 128),
                "dim2": np.linspace(0, 1, 64),
            }}},
        "code": {"name": "eq_analyze"}}

    TimeSlice = EquilibriumTimeSlice

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    time_slice: TimeSeriesAoS[EquilibriumTimeSlice] = sp_property(coordinate1="time", type="dynamic")

    def refresh(self, *args, **kwargs):
        """update the last time slice"""

        self.time_slice.refresh(*args, **kwargs)

        # self.grids_ggd.refresh(*args, **kwargs)

    def advance(self, *args, **kwargs):
        self.time_slice.advance(*args, **kwargs)
        # self.grids_ggd.advance(*args, **kwargs)

    def __geometry__(self, view="RZ", **kwargs) -> GeoObject:
        return self.time_slice.current.__geometry__(view=view, **kwargs)
