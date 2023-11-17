from __future__ import annotations
import numpy as np

from spdm.data.AoS import AoS
from spdm.data.Expression import Expression
from spdm.data.sp_property import sp_property
from spdm.data.TimeSeries import TimeSeriesAoS, TimeSlice
from spdm.geometry.Curve import Curve
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Point import Point
from spdm.utils.tags import _not_found_
from spdm.data.Path import update_tree
from spdm.mesh.Mesh import Mesh

from .Utilities import *

from ..utils.logger import logger

from ..ontology import equilibrium


@sp_tree(mesh="grid")
class EquilibriumCoordinateSystem(equilibrium._T_equilibrium_coordinate_system):
    grid_type: Identifier

    grid: Mesh

    radial_grid: CoreRadialGrid

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


@sp_tree(coordinate1="psi", extrapolate="zeros")
class EquilibriumProfiles1D(equilibrium._T_equilibrium_profiles_1d):
    """
    1D profiles of the equilibrium quantities
    NOTE:
        - psi_norm is the normalized poloidal flux
        - psi is the poloidal flux,
        - 以psi而不是psi_norm为主坐标,原因是 profiles1d 中涉及对 psi 的求导和积分
    """

 
    @property
    def _root(self) -> Equilibrium.TimeSlice:
        return self._parent

    @sp_property
    def grid(self) -> CoreRadialGrid:
        return CoreRadialGrid(
            {
                "psi_norm": self.psi_norm,
                "psi_axis": self._root.global_quantities.psi_axis,
                "psi_boundary": self._root.global_quantities.psi_boundary,
                "rho_tor_norm": self.rho_tor_norm,
                "rho_tor_boundary": self.rho_tor(self._root.global_quantities.psi_boundary),
            }
        )

    psi_norm: array_type = sp_property(units="-")

    psi: Expression = sp_property(units="Wb")

    dphi_dpsi: Expression = sp_property(label=r"\frac{d\phi}{d\psi}", units="-")

    phi: Expression = sp_property(units="Wb",label=r"\phi")

    pressure: Expression = sp_property(units="Pa",label=r"P")

    f: Expression = sp_property(units="T.m")

    dpressure_dpsi: Expression = sp_property(units="Pa.Wb^-1")

    f_df_dpsi: Expression = sp_property(units="T^2.m^2/Wb", label=r"\frac{f d f}{d \psi}")

    j_tor: Expression = sp_property(units="A.m^-2")

    j_parallel: Expression = sp_property(units="A/m^2")

    q: Expression

    magnetic_shear: Expression

    r_inboard: Expression = sp_property(units="m")

    r_outboard: Expression = sp_property(units="m")

    rho_tor: Expression = sp_property(units="m",label=r"\rho_{tor}")

    rho_tor_norm: Expression= sp_property(units="m",label=r"\bar{\rho_{tor}}")

    dpsi_drho_tor: Expression = sp_property(units="Wb/m",label=r"\frac{d\psi}{\rho_{tor}}")

    @sp_property
    def geometric_axis(self) -> RZTuple:
        return {"r": self.major_radius, "z": self.magnetic_z}

    minor_radius: Expression = sp_property(units="m")

    major_radius: Expression = sp_property(units="m")  # R0

    magnetic_z: Expression = sp_property(units="m")  # Z0

    elongation: Expression

    triangularity_upper: Expression

    triangularity_lower: Expression

    @sp_property
    def triangularity(self) -> Expression:
        return (self.triangularity_upper + self.triangularity_lower) * 0.5

    squareness_upper_inner: Expression

    squareness_upper_outer: Expression

    squareness_lower_inner: Expression

    squareness_lower_outer: Expression

    squareness: Expression = sp_property(default_value=1.0)

    volume: Expression = sp_property(units="m^3")

    rho_volume_norm: Expression

    dvolume_dpsi: Expression = sp_property(units="m^3.Wb^-1")

    dvolume_drho_tor: Expression = sp_property(units="m^2", label=r"\frac{dvolume}{d\rho_{tor}}")

    area: Expression = sp_property(units="m^2")

    darea_dpsi: Expression = sp_property(units="m^2.Wb^-1")

    darea_drho_tor: Expression = sp_property(units="m")

    surface: Expression = sp_property(units="m^2")

    trapped_fraction: Expression

    gm1: Expression
    gm2: Expression
    gm3: Expression
    gm4: Expression
    gm5: Expression
    gm6: Expression
    gm7: Expression
    gm8: Expression
    gm9: Expression

    b_field_average: Expression = sp_property(units="T")

    b_field_min: Expression = sp_property(units="T")

    b_field_max: Expression = sp_property(units="T")

    beta_pol: Expression

    mass_density: Expression = sp_property(units="kg.m^-3")


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

    outline: Curve

    psi_norm: float

    psi: float = sp_property(units="Wb")

    geometric_axis: Point

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

    x_point: AoS[Point]

    strike_point: AoS[Point]

    active_limiter_point: Point


@sp_tree
class EquilibriumBoundarySeparatrix(equilibrium._T_equilibrium_boundary_separatrix):
    type: int

    outline: CurveRZ

    psi: float = sp_property(units="Wb")

    geometric_axis: Point

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

    x_point: AoS[Point]

    strike_point: AoS[Point]

    active_limiter_point: Point


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

    boundary: EquilibriumBoundary

    boundary_separatrix: BoundarySeparatrix

    constraints: Constraints

    global_quantities: EquilibriumGlobalQuantities

    profiles_1d: Profiles1D

    profiles_2d: Profiles2D

    coordinate_system: CoordinateSystem

    ggd: GGD

    # def __geometry__(self, view_port="RZ", **kwargs) -> GeoObject:
    #     geo = {}

    #     if view_port == "RZ":
    #         o_points, x_points = self.coordinate_system.critical_points

    #         geo["o_points"] = [Point(p.r, p.z, name=f"{idx}") for idx, p in enumerate(o_points)]
    #         geo["x_points"] = [Point(p.r, p.z, name=f"{idx}") for idx, p in enumerate(x_points)]

    #         geo["boundary"] = Curve(self.boundary.outline.r.__array__(), self.boundary.outline.z.__array__())

    #         geo["boundary_separatrix"] = Curve(
    #             self.boundary_separatrix.outline.r.__array__(),
    #             self.boundary_separatrix.outline.z.__array__(),
    #         )

    #     geo["psi"] = self.profiles_2d.psi.__geometry__()

    #     styles = {
    #         "o_points": {"$matplotlib": {"c": "red", "marker": "."}},
    #         "x_points": {"$matplotlib": {"c": "blue", "marker": "x"}},
    #         "boundary": {"$matplotlib": {"color": "red", "linewidth": 0.5}},
    #         "boundary_separatrix": {
    #             "$matplotlib": {
    #                 "color": "red",
    #                 "linestyle": "dashed",
    #                 "linewidth": 0.25,
    #             }
    #         },
    #     }
    #     styles = update_tree(styles, kwargs)

    #     return geo, styles

    def __geometry__(self, view_point="RZ", **kwargs) -> GeoObject:
        """
        plot o-point,x-point,lcfs,separatrix and contour of psi
        """

        geo = {}
        styles = {}

        match view_point.lower():
            case "rz":
                geo["o_points"] = Point(self.global_quantities.magnetic_axis.r, self.global_quantities.magnetic_axis.z)

                geo["x_points"] = [Point(p.r, p.z, name=f"{idx}") for idx, p in enumerate(self.boundary.x_point)]

                geo["strike_points"] = [
                    Point(p.r, p.z, name=f"{idx}") for idx, p in enumerate(self.boundary.strike_point)
                ]

                geo["boundary"] = self.boundary.outline

                geo["boundary_separatrix"] = self.boundary_separatrix.outline

                geo["psi"], styles["psi"] = self.profiles_2d.psi.__geometry__()

                styles["o_points"] = {"$matplotlib": {"color": "red", "marker": ".", "linewidths": 0.5}}
                styles["x_points"] = {"$matplotlib": {"color": "blue", "marker": "x", "linewidths": 0.5}}
                styles["boundary"] = {"$matplotlib": {"color": "blue", "linestyle": "dotted", "linewidth": 0.5}}
                styles["boundary_separatrix"] = {
                    "$matplotlib": {"color": "red", "linestyle": "dashed", "linewidth": 0.25}
                }
                styles["psi"].update({"$matplotlib": {"levels": 40, "cmap": "jet"}})

        styles = update_tree(styles, kwargs)

        return geo, styles


@sp_tree
class Equilibrium(Module):
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

    _plugin_prefix = "fytok.plugins.equilibrium."

    _metadata = {"code": {"name": "fy_eq"}}  # default plugin

    ids_properties: IDSProperties

    TimeSlice = EquilibriumTimeSlice

    time_slice: TimeSeriesAoS[EquilibriumTimeSlice]

    def __geometry__(self, *args, **kwargs):
        return self.time_slice.current.__geometry__(*args, **kwargs)
