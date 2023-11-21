from scipy import constants
from copy import copy
from fytok.utils.logger import logger
from spdm.data.Expression import Expression
from spdm.data.sp_property import sp_tree, sp_property
from spdm.data.TimeSeries import TimeSlice
from spdm.utils.tags import _not_found_
from spdm.data.Entry import Entry
from spdm.utils.typing import array_type, as_array
from .CoreProfiles import CoreProfiles
from .CoreSources import CoreSources
from .CoreTransport import CoreTransport
from .Equilibrium import Equilibrium
from .Utilities import *
from ..ontology import transport_solver_numerics

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * constants.pi


@sp_tree
class TransportSolverNumericsEquationPrimary:
    """Profile and derivatives a the primary quantity for a 1D transport equation"""

    identifier: str
    """ Identifier of the primary quantity of the transport equation. The description
        node contains the path to the quantity in the physics IDS (example:
        core_profiles/profiles_1d/ion/D/density)"""
    profile: array_type = 0.0
    """ Profile of the primary quantity"""

    flux: array_type
    """ Flux of the primary quantity"""

    d_dr: Expression | array_type
    """ Radial derivative with respect to the primary coordinate"""

    dflux_dr: Expression | array_type
    """ Radial derivative of Flux of the primary quantity"""

    d2_dr2: Expression | array_type
    """ Second order radial derivative with respect to the primary coordinate"""

    d_dt: Expression | array_type
    """ Time derivative"""

    d_dt_cphi: Expression | array_type
    """ Derivative with respect to time, at constant toroidal flux (for current
        diffusion equation)"""

    d_dt_cr: Expression | array_type
    """ Derivative with respect to time, at constant primary coordinate coordinate (for
        current diffusion equation)"""


@sp_tree
class TransportSolverNumericsEquation:
    primary_quantity: TransportSolverNumericsEquationPrimary
    """ Profile and derivatives of the primary quantity of the transport equation"""

    @sp_tree
    class EquationBC:
        identifier: Identifier
        """ Identifier of the boundary condition type.  ID =
            1: value of the field y;
            2: radial derivative of the field (-dy/drho_tor);
            3: scale length of the field y/(-dy/drho_tor);
            4: flux;
            5: generic boundary condition y expressed as a1y'+a2y=a3.
            6: equation not solved;
        """
        value: array_type
        """ Value of the boundary condition.
            For ID = 1 to 4, only the first position in the vector is used.
            For ID = 5, all three positions are used, meaning respectively a1, a2, a3.
        """
        rho_tor_norm: float
        """ Position, in normalised toroidal flux, at which the boundary condition is
        imposed. Outside this position, the value of the data are considered to be
        prescribed."""

        func: Expression

    boundary_condition: AoS[EquationBC]

    coefficient: AoS
    """ Set of numerical coefficients involved in the transport equation"""

    convergence: PropertyTree
    """ Convergence details"""


@sp_tree(coordinate1="grid/rho_tor_norm")
class TransportSolverNumericsTimeSlice(TimeSlice):
    """Numerics related to 1D radial solver for a given time slice"""

    Equation = TransportSolverNumericsEquation

    grid: CoreRadialGrid
    """ Radial grid"""

    equation: AoS[TransportSolverNumericsEquation]
    """ Set of transport equations"""

    control_parameters: PropertyTree
    """ Solver-specific input or output quantities"""

    drho_tor_dt: Expression = sp_property(units="m.s^-1")
    """ Partial derivative of the toroidal flux coordinate profile with respect to time"""

    d_dvolume_drho_tor_dt: Expression = sp_property(units="m^2.s^-1")
    """ Partial derivative with respect to time of the derivative of the volume with
      respect to the toroidal flux coordinate"""


@sp_tree
class TransportSolverNumericsBC:
    """Boundary conditions of radial transport equations for a given time slice"""

    @sp_tree
    class ParticleBC:
        particles: array_type
        """ Particle flux"""

        energy: array_type
        """ Energy flux"""

        momentum: array_type
        """ Momentum flux"""

    rho_tor_norm: float
    """ Position, in normalised toroidal flux, at which the boundary condition is
       imposed. Outside this position, the value of the data are considered to be
       prescribed."""

    identifier: Identifier
    """ Identifier of the boundary condition type. ID = 1: poloidal flux; 2: ip; 3: loop
                voltage; 4: undefined; 5: generic boundary condition y expressed as a1y'+a2y=a3.
                6: equation not solved;"""

    current: float
    """ Boundary condition for the current diffusion equation."""

    electrons: ParticleBC
    """ Quantities related to the electrons"""

    ion: AoS[ParticleBC]
    """ Quantities related to the different ion species"""

    energy_ion_total: ParticleBC = sp_property(units="W.m^-3")
    """ Boundary condition for the ion total (sum over ion species) energy equation
       (temperature if ID = 1)"""

    momentum_tor: ParticleBC = sp_property(units="kg.m.s^-1")
    """ Boundary condition for the total plasma toroidal momentum equation (summed over
       ion species and electrons) (momentum if ID = 1)"""


@sp_tree
class TransportSolverNumerics(IDS):
    r"""Solve transport equations  $\rho=\sqrt{ \Phi/\pi B_{0}}$"""

    _plugin_prefix = "fytok.plugins.transport_solver_numerics."

    code: Code = {"name": None}

    solver: Identifier

    primary_coordinate: Identifier = "rho_tor_norm"  # $\rho_{tor}=\sqrt{ \Phi/\pi B_{0}}$

    equations: AoS[TransportSolverNumericsEquation]

    TimeSlice = TransportSolverNumericsTimeSlice

    time_slice: TimeSeriesAoS[TransportSolverNumericsTimeSlice]

    def parser_arguments(self, *args, **kwargs) -> typing.Tuple[typing.Any]:
        args, kwargs = super().parser_arguments(*args, **kwargs)

        equilibrium: Equilibrium = self._inputs["equilibrium"]

        rho_tor_norm = kwargs.pop("rho_tor_norm", _not_found_)

        if rho_tor_norm is _not_found_:
            rho_tor_norm = self.code.parameters.get("rho_tor_norm", None)

        previous = self.time_slice.previous

        if previous is not _not_found_:
            grid = previous.grid.duplicate(rho_tor_norm)

        else:
            grid = equilibrium.time_slice.current.profiles_1d.grid.duplicate(rho_tor_norm)

        # equation = [
        #     {
        #         "primary_quantity": {
        #             "identifier": equ.primary_quantity.identifier,
        #             "profile": equ.primary_quantity.profile,
        #         },
        #         "boundary_condition": [
        #             {
        #                 "identifier": equ.boundary_condition[0].identifier,
        #                 "value": equ.boundary_condition[0].value,
        #             },
        #             {
        #                 "identifier": equ.boundary_condition[1].identifier,
        #                 "value": equ.boundary_condition[1].value,
        #             },
        #         ],
        #     }
        #     for equ in equation_s
        # ]

        return [*args, {"grid": grid}], kwargs

    def execute(self, current: TimeSlice, *args, **kwargs):
        pass

    def refresh(
        self,
        *args,
        equilibrium: Equilibrium = None,
        core_transport: CoreTransport = None,
        core_sources: CoreSources = None,
        **kwargs,
    ):
        super().refresh(
            *args,
            equilibrium=equilibrium,
            core_transport=core_transport,
            core_sources=core_sources,
            **kwargs,
        )

    def advance(
        self,
        *args,
        equilibrium: Equilibrium = None,
        core_transport: CoreTransport = None,
        core_sources: CoreSources = None,
        **kwargs,
    ):
        super().advance(
            *args,
            equilibrium=equilibrium,
            core_sources=core_sources,
            core_transport=core_transport,
            **kwargs,
        )
