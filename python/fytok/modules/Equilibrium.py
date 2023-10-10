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
from spdm.mesh.Mesh import Mesh
from ..utils.utilities import *
from ..utils.logger import logger


@sp_tree(mesh="../grid")
class EquilibriumCoordinateSystem:
    """Flux surface coordinate system on a square grid of flux and poloidal angle """

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    grid_type: Identifier

    grid: Mesh

    r: Field = sp_property(units="m")

    z: Field = sp_property(units="m")

    jacobian: Field = sp_property(units="mixed")

    tensor_covariant: array_type = sp_property(coordinate3="1...3", coordinate4="1...3", units="mixed")

    tensor_contravariant: array_type = sp_property(coordinate3="1...3", coordinate4="1...3", units="mixed")


@sp_tree
class EquilibriumGlobalQuantities:

    beta_pol: float = sp_property(units="-")

    beta_tor: float = sp_property(units="-")

    beta_normal: float = sp_property(units="-")

    ip: float = sp_property(units="A")

    li_3: float = sp_property(units="-")

    volume: float = sp_property(units="m^3")

    area: float = sp_property(units="m^2")

    surface: float = sp_property(units="m^2")

    length_pol: float = sp_property(units="m")

    psi_axis: float = sp_property(units="Wb")

    psi_boundary: float = sp_property(units="Wb")

    @sp_tree
    class MagneticAxis:
        r: float = sp_property(units="m")
        z: float = sp_property(units="m")
        b_field_tor: float = sp_property(units="T")

    magnetic_axis: MagneticAxis = sp_property()
    """ Magnetic axis position and toroidal field"""

    @sp_tree
    class CurrentCentre:
        r: float = sp_property(units="m")
        z: float = sp_property(units="m")
        velocity_z: float = sp_property(units="m.s^-1")

    current_centre: CurrentCentre = sp_property()

    q_axis: float = sp_property(units="-")

    q_95: float = sp_property(units="-")

    @sp_tree
    class Qmin:
        value: float = sp_property(units="-")
        rho_tor_norm: float = sp_property(units="-")

    q_min: Qmin = sp_property()

    energy_mhd: float = sp_property(units="J")

    psi_external_average: float = sp_property(units="Wb")

    v_external: float = sp_property(units="V")

    plasma_inductance: float = sp_property(units="H")

    plasma_resistance: float = sp_property(units="ohm")


@sp_tree(coordinate1="../psi")
class EquilibriumProfiles1D:

    psi: array_type = sp_property(units="Wb")

    phi: Function = sp_property(units="Wb")

    pressure: Function = sp_property(units="Pa")

    f: Function = sp_property(units="T.m")

    dpressure_dpsi: Function = sp_property(units="Pa.Wb^-1")

    f_df_dpsi: Function = sp_property(units="T^2.m^2/Wb")

    j_tor: Function = sp_property(units="A.m^-2")

    j_parallel: Function = sp_property(units="A/m^2")

    q: Function = sp_property(units="-")

    magnetic_shear: Function = sp_property(units="-")

    r_inboard: Function = sp_property(units="m")

    r_outboard: Function = sp_property(units="m")

    rho_tor: Function = sp_property(units="m")

    rho_tor_norm: Function = sp_property(units="-")

    dpsi_drho_tor: Function = sp_property(units="Wb/m")

    @sp_tree
    class RZ:
        r: Function
        z: Function

    geometric_axis: RZ

    elongation: Function = sp_property(units="-")

    triangularity_upper: Function = sp_property(units="-")

    triangularity_lower: Function = sp_property(units="-")

    squareness_upper_inner: Function = sp_property(units="-")

    squareness_upper_outer: Function = sp_property(units="-")

    squareness_lower_inner: Function = sp_property(units="-")

    squareness_lower_outer: Function = sp_property(units="-")

    volume: Function = sp_property(units="m^3")

    rho_volume_norm: Function = sp_property(units="-")

    dvolume_dpsi: Function = sp_property(units="m^3.Wb^-1")

    dvolume_drho_tor: Function = sp_property(units="m^2")

    area: Function = sp_property(units="m^2")

    darea_dpsi: Function = sp_property(units="m^2.Wb^-1")

    darea_drho_tor: Function = sp_property(units="m")

    surface: Function = sp_property(units="m^2")

    trapped_fraction: Function = sp_property(units="-")

    gm1: Function = sp_property(units="m^-2")
    gm2: Function = sp_property(units="m^-2")
    gm3: Function = sp_property(units="-")
    gm4: Function = sp_property(units="T^-2")
    gm5: Function = sp_property(units="T^2")
    gm6: Function = sp_property(units="T^-2")
    gm7: Function = sp_property(units="-")
    gm8: Function = sp_property(units="m")
    gm9: Function = sp_property(units="m^-1")

    b_field_average: Function = sp_property(units="T")

    b_field_min: Function = sp_property(units="T")

    b_field_max: Function = sp_property(units="T")

    beta_pol: Function = sp_property(units="-")

    mass_density: Function = sp_property(units="kg.m^-3")

    minor_radius: Function = sp_property(units="m")

    triangularity: Function = sp_property(units="-")

    squareness: Function = sp_property(units="-")


@sp_tree(mesh="grid")
class EquilibriumProfiles2D:

    type: Identifier = sp_property()

    grid_type: Identifier

    grid: Mesh

    r: Field = sp_property(units="m")

    z: Field = sp_property(units="m")

    psi: Field = sp_property(units="Wb")

    theta: Field = sp_property(units="rad")

    phi: Field = sp_property(units="Wb")

    j_tor: Field = sp_property(units="A.m^-2")

    j_parallel: Field = sp_property(units="A.m^-2")

    b_field_r: Field = sp_property(units="T")

    b_field_z: Field = sp_property(units="T")

    b_field_tor: Field = sp_property(units="T")


@sp_tree
class EquilibriumBoundary:

    type: int = sp_property()

    outline: CurveRZ = sp_property()

    psi_norm: float = sp_property(units="-")

    psi: float = sp_property(units="Wb")

    geometric_axis: PointRZ = sp_property()

    type: int = sp_property()

    outline: CurveRZ = sp_property()

    psi_norm: float = sp_property(units="-")

    psi: float = sp_property(units="Wb")

    geometric_axis: PointRZ = sp_property()

    minor_radius: float = sp_property(units="m")

    elongation: float = sp_property(units="-")

    elongation_upper: float = sp_property(units="-")

    elongation_lower: float = sp_property(units="-")

    triangularity: float = sp_property(units="-")

    triangularity_upper: float = sp_property(units="-")

    triangularity_lower: float = sp_property(units="-")

    squareness_upper_inner: float = sp_property(units="-")

    squareness_upper_outer: float = sp_property(units="-")

    squareness_lower_inner: float = sp_property(units="-")

    squareness_lower_outer: float = sp_property(units="-")

    x_point: AoS[PointRZ] = sp_property(coordinate1="1...N")

    strike_point: AoS[PointRZ] = sp_property(coordinate1="1...N")

    active_limiter_point: PointRZ = sp_property()


@sp_tree
class EquilibriumBoundarySeparatrix:

    type: int = sp_property()

    outline: CurveRZ = sp_property()

    psi: float = sp_property(units="Wb")

    geometric_axis: PointRZ = sp_property()

    minor_radius: float = sp_property(units="m")

    elongation: float = sp_property(units="-")

    elongation_upper: float = sp_property(units="-")

    elongation_lower: float = sp_property(units="-")

    triangularity: float = sp_property(units="-")

    triangularity_upper: float = sp_property(units="-")

    triangularity_lower: float = sp_property(units="-")

    squareness_upper_inner: float = sp_property(units="-")

    squareness_upper_outer: float = sp_property(units="-")

    squareness_lower_inner: float = sp_property(units="-")

    squareness_lower_outer: float = sp_property(units="-")

    x_point: AoS[PointRZ] = sp_property(coordinate1="1...N")

    strike_point: AoS[PointRZ] = sp_property(coordinate1="1...N")

    active_limiter_point: PointRZ = sp_property()

    @sp_tree
    class BoundaryClosest:
        r: float
        z: float
        distance: float

    closest_wall_point: BoundaryClosest = sp_property()

    dr_dz_zero_point: PointRZ = sp_property()

    @sp_tree
    class Gap:
        name: str = sp_property()
        identifier: str = sp_property()
        r: float = sp_property(units="m")
        z: float = sp_property(units="m")
        angle: float = sp_property(units="rad")
        value: float = sp_property(units="m")

    gap: AoS[Gap]


@sp_tree
class EequilibriumConstraints:
    pass


@sp_tree
class EquilibriumTimeSlice(TimeSlice):

    Constraints = EequilibriumConstraints
    BoundarySeparatrix = EquilibriumBoundarySeparatrix
    Boundary = EquilibriumBoundary
    GlobalQuantities = EquilibriumGlobalQuantities
    CoordinateSystem = EquilibriumCoordinateSystem
    Profiles1D = EquilibriumProfiles1D
    Profiles2D = EquilibriumProfiles2D

    @sp_tree
    class VacuumToroidalField:
        r0: float
        b0: float

    vacuum_toroidal_field: VacuumToroidalField

    boundary: Boundary

    boundary_separatrix: BoundarySeparatrix

    constraints: Constraints

    global_quantities: GlobalQuantities

    profiles_1d: Profiles1D

    profiles_2d: AoS[Profiles2D]

    coordinate_system: CoordinateSystem

    def __geometry__(self, view_port="RZ", **kwargs) -> GeoObject:
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


class Equilibrium(IDS):
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

    time_slice: TimeSeriesAoS[EquilibriumTimeSlice] = sp_property(coordinate1="time", )

    def refresh(self, *args, **kwargs):
        """update the last time slice"""

        self.time_slice.refresh(*args, **kwargs)

        # self.grids_ggd.refresh(*args, **kwargs)

    def advance(self, *args, **kwargs):
        self.time_slice.advance(*args, **kwargs)
        # self.grids_ggd.advance(*args, **kwargs)

    def __geometry__(self,  *args,  **kwargs) -> GeoObject:
        return self.time_slice.current.__geometry__(*args, **kwargs)
