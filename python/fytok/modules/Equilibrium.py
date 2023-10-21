from __future__ import annotations
import numpy as np

from spdm.data.AoS import AoS
from spdm.data.Function import Function
from spdm.data.sp_property import sp_property
from spdm.data.TimeSeries import TimeSeriesAoS, TimeSlice
from spdm.geometry.Curve import Curve
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Point import Point
from spdm.utils.tags import _not_found_
from spdm.utils.tree_utils import merge_tree_recursive
from spdm.mesh.Mesh import Mesh

from .Utilities import *

from ..utils.logger import logger

from ..ontology import equilibrium


@sp_tree(mesh="../grid")
class EquilibriumCoordinateSystem(equilibrium._T_equilibrium_coordinate_system):

    grid_type: Identifier

    grid: Mesh

    r: Field = sp_property(units="m")

    z: Field = sp_property(units="m")

    jacobian: Field = sp_property(units="mixed")

    tensor_covariant: array_type = sp_property(coordinate3="1...3", coordinate4="1...3", units="mixed")

    tensor_contravariant: array_type = sp_property(coordinate3="1...3", coordinate4="1...3", units="mixed")


@sp_tree
class EquilibriumGlobalQuantities(equilibrium._T_equilibrium_global_quantities):

    beta_pol: float

    beta_tor: float

    beta_normal: float

    ip: float = sp_property(units="A")

    li_3: float

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

    magnetic_axis: MagneticAxis
    """ Magnetic axis position and toroidal field"""

    @sp_tree
    class CurrentCentre:
        r: float = sp_property(units="m")
        z: float = sp_property(units="m")
        velocity_z: float = sp_property(units="m.s^-1")

    current_centre: CurrentCentre

    q_axis: float

    q_95: float

    @sp_tree
    class Qmin:
        value: float
        rho_tor_norm: float

    q_min: Qmin

    energy_mhd: float = sp_property(units="J")

    psi_external_average: float = sp_property(units="Wb")

    v_external: float = sp_property(units="V")

    plasma_inductance: float = sp_property(units="H")

    plasma_resistance: float = sp_property(units="ohm")


@sp_tree(coordinate1="psi", extrapolate='zeros')
class EquilibriumProfiles1D(equilibrium._T_equilibrium_profiles_1d):

    @sp_property
    def grid(self) -> CoreRadialGrid:
        g = self._parent.global_quantities
        return CoreRadialGrid({
            "psi_norm": (self.psi-g .psi_boundary)/(g .psi_axis-g .psi_boundary),
            "rho_tor_norm": self.rho_tor_norm(self.psi),
            "psi_magnetic_axis": g .psi_axis,
            "psi_boundary": g .psi_boundary,
            "rho_tor_boundary": self.rho_tor(g .psi_boundary),
        })

    psi: array_type = sp_property(units="Wb")

    psi_norm: Function = sp_property(units="-")

    phi: Function = sp_property(units="Wb")

    pressure: Function = sp_property(units="Pa")

    f: Function = sp_property(units="T.m")

    dpressure_dpsi: Function = sp_property(units="Pa.Wb^-1")

    f_df_dpsi: Function = sp_property(units="T^2.m^2/Wb")

    j_tor: Function = sp_property(units="A.m^-2")

    j_parallel: Function = sp_property(units="A/m^2")

    q: Function

    magnetic_shear: Function

    r_inboard: Function = sp_property(units="m")

    r_outboard: Function = sp_property(units="m")

    rho_tor: Function = sp_property(units="m")

    rho_tor_norm: Function

    dpsi_drho_tor: Function = sp_property(units="Wb/m")

    @sp_property
    def geometric_axis(self) -> RZTuple: return {"r": self.major_radius,  "z":  self.magnetic_z}

    minor_radius: Function = sp_property(units="m")

    major_radius: Function = sp_property(units="m")  # R0

    magnetic_z: Function = sp_property(units="m")  # Z0

    elongation: Function

    triangularity_upper: Function

    triangularity_lower: Function

    @sp_property
    def triangularity(self) -> Function: return (self.triangularity_upper+self.triangularity_lower)*0.5

    squareness_upper_inner: Function

    squareness_upper_outer: Function

    squareness_lower_inner: Function

    squareness_lower_outer: Function

    squareness: Function = sp_property(default_value=1.0)

    volume: Function = sp_property(units="m^3")

    rho_volume_norm: Function

    dvolume_dpsi: Function = sp_property(units="m^3.Wb^-1")

    dvolume_drho_tor: Function = sp_property(units="m^2")

    area: Function = sp_property(units="m^2")

    darea_dpsi: Function = sp_property(units="m^2.Wb^-1")

    darea_drho_tor: Function = sp_property(units="m")

    surface: Function = sp_property(units="m^2")

    trapped_fraction: Function

    gm1: Function = sp_property(units="m^-2")
    gm2: Function = sp_property(units="m^-2")
    gm3: Function
    gm4: Function = sp_property(units="T^-2")
    gm5: Function = sp_property(units="T^2")
    gm6: Function = sp_property(units="T^-2")
    gm7: Function
    gm8: Function = sp_property(units="m")
    gm9: Function = sp_property(units="m^-1")

    b_field_average: Function = sp_property(units="T")

    b_field_min: Function = sp_property(units="T")

    b_field_max: Function = sp_property(units="T")

    beta_pol: Function

    mass_density: Function = sp_property(units="kg.m^-3")


@sp_tree(mesh="grid")
class EquilibriumProfiles2D(equilibrium._T_equilibrium_profiles_2d):

    type: Identifier

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
class EquilibriumBoundary(equilibrium._T_equilibrium_boundary):

    type: int

    outline: CurveRZ

    psi_norm: float

    psi: float = sp_property(units="Wb")

    geometric_axis: PointRZ

    minor_radius: float = sp_property(units="m")

    elongation: float

    elongation_upper: float

    elongation_lower: float

    triangularity: float

    triangularity_upper: float

    triangularity_lower: float

    squareness_upper_inner: float

    squareness_upper_outer: float

    squareness_lower_inner: float

    squareness_lower_outer: float

    x_point: AoS[PointRZ]

    strike_point: AoS[PointRZ]

    active_limiter_point: PointRZ


@sp_tree
class EquilibriumBoundarySeparatrix(equilibrium._T_equilibrium_boundary_separatrix):

    type: int

    outline: CurveRZ

    psi: float = sp_property(units="Wb")

    geometric_axis: PointRZ

    minor_radius: float = sp_property(units="m")

    elongation: float

    elongation_upper: float

    elongation_lower: float

    triangularity: float

    triangularity_upper: float

    triangularity_lower: float

    squareness_upper_inner: float

    squareness_upper_outer: float

    squareness_lower_inner: float

    squareness_lower_outer: float

    x_point: AoS[PointRZ]

    strike_point: AoS[PointRZ]

    active_limiter_point: PointRZ


@sp_tree
class EequilibriumConstraints(equilibrium._T_equilibrium_constraints):
    pass


@sp_tree
class EquilibriumGGD(equilibrium._T_equilibrium_ggd):
    pass


@sp_tree
class EquilibriumTimeSlice(equilibrium._T_equilibrium_time_slice):

    Constraints = EequilibriumConstraints
    BoundarySeparatrix = EquilibriumBoundarySeparatrix
    Boundary = EquilibriumBoundary
    GlobalQuantities = EquilibriumGlobalQuantities
    CoordinateSystem = EquilibriumCoordinateSystem
    Profiles1D = EquilibriumProfiles1D
    Profiles2D = EquilibriumProfiles2D
    GGD = EquilibriumGGD

    vacuum_toroidal_field: VacuumToroidalField

    boundary: Boundary

    boundary_separatrix: BoundarySeparatrix

    constraints: Constraints

    global_quantities: GlobalQuantities

    profiles_1d: Profiles1D

    profiles_2d: Profiles2D

    coordinate_system: CoordinateSystem

    ggd: GGD

    def __geometry__(self, view_port="RZ", **kwargs) -> GeoObject:
        geo = {}

        if view_port == "RZ":
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


class Equilibrium(TimeBasedActor):
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

    _metadata = {"code": {"name": "eq_analyze"}}

    ids_properties: IDSProperties

    TimeSlice = EquilibriumTimeSlice

    time_slice: TimeSeriesAoS[EquilibriumTimeSlice]

    def __geometry__(self,  *args,  **kwargs):
        return self.time_slice.current.__geometry__(*args, **kwargs)
