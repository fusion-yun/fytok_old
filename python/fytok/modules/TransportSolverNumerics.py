from scipy import constants
from fytok.utils.logger import logger
from spdm.data.sp_property import sp_tree, sp_property
from spdm.data.TimeSeries import TimeSlice
from spdm.utils.tags import _not_found_
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

    identifier: Identifier
    """ Identifier of the primary quantity of the transport equation. The description
        node contains the path to the quantity in the physics IDS (example:
        core_profiles/profiles_1d/ion(1)/density)"""

    profile: array_type
    """ Profile of the primary quantity"""

    d_dr: array_type
    """ Radial derivative with respect to the primary coordinate"""

    d2_dr2: array_type
    """ Second order radial derivative with respect to the primary coordinate"""

    d_dt: array_type
    """ Time derivative"""

    d_dt_cphi: array_type
    """ Derivative with respect to time, at constant toroidal flux (for current
        diffusion equation)"""

    d_dt_cr: array_type
    """ Derivative with respect to time, at constant primary coordinate coordinate (for
        current diffusion equation)"""


@sp_tree
class TransportSolverNumericsEquation:

    computation_mode: int
    """
        Name          Index    Description
        static            0    Equation is not solved, no profile evolution
        interpretative    1    Equation is not solved, profile is evolved by interpolating from input objects
        predictive        2    Equation is solved, profile evolves
    """

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

    primary_quantity: TransportSolverNumericsEquationPrimary
    """ Profile and derivatives of the primary quantity of the transport equation"""

    boundary_condition: AoS[EquationBC]
    """ Set of boundary conditions of the transport equation"""

    coefficient: AoS
    """ Set of numerical coefficients involved in the transport equation"""

    convergence: AttributeTree
    """ Convergence details"""


@sp_tree(coordinate1="grid/rho_tor_norm")
class TransportSolverNumericsSolver1D:
    """Numerics related to 1D radial solver for a given time slice"""
    Equation = TransportSolverNumericsEquation

    grid: CoreRadialGrid
    """ Radial grid"""

    equation: AoS[TransportSolverNumericsEquation]
    """ Set of transport equations"""

    control_parameters: AttributeTree
    """ Solver-specific input or output quantities"""

    drho_tor_dt: Function = sp_property(units="m.s^-1")
    """ Partial derivative of the toroidal flux coordinate profile with respect to time"""

    d_dvolume_drho_tor_dt: Function = sp_property(units="m^2.s^-1")
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
class TransportSolverNumericsSlice(TimeSlice):

    primary_coordinate: Identifier

    vacuum_toroidal_field: VacuumToroidalField

    Solver1D = TransportSolverNumericsSolver1D
    solver_1d: TransportSolverNumericsSolver1D
    """ Numerics related to 1D radial solver, for various time slices."""

    boundary_conditions_ggd: transport_solver_numerics._T_numerics_bc_ggd
    """ Boundary conditions of the transport equations, provided on the GGD, for various
        time slices"""

    convergence: transport_solver_numerics._T_numerics_convergence
    """ Convergence details To be removed when the solver_1d structure is finalized."""


@sp_tree
class TransportSolverNumerics(Module):
    r"""Solve transport equations
        :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`
    """

    _plugin_prefix = 'fytok.plugins.transport_solver_numerics.'

    solver: Identifier
    """ Solver identifier"""

    TimeSlice = TransportSolverNumericsSlice

    time_slice: TimeSeriesAoS[TransportSolverNumericsSlice]

    def refresh(self, *args,
                core_profiles: CoreProfiles,
                equilibrium: Equilibrium,
                core_transport: CoreTransport.Model,
                core_sources: CoreSources.Source,
                **kwargs):
        """
            solve transport equation until residual < tolerance
            return core_profiles

        """

        core_profiles_1d = core_profiles.time_slice.current.profiles_1d

        idx = 0
        # fmt:off
        equations = [
            # {"primary_quantity":{"identifier":{"name": "psi",                               "index":(idx       ) }}, "boundary_conditions": []},

            {"primary_quantity":{"identifier":{"name": "electrons/density_thermal",         "index":(idx:=idx+1) }}, "boundary_conditions": []},
            # {"primary_quantity":{"identifier":{"name": "electrons/density_fast",            "index":(idx:=idx+1) }}, "boundary_conditions": []},
            # {"primary_quantity":{"identifier":{"name": "electrons/temperature",             "index":(idx:=idx+1) }}, "boundary_conditions": []},
            # {"primary_quantity":{"identifier":{"name": "electrons/momentum",                "index":(idx:=idx+1) }}, "boundary_conditions": []},
            *sum([[
            {"primary_quantity":{"identifier":{"name": f"ion/{s}/density_thermal",          "index":(idx:=idx+1) }}, "boundary_conditions": []},
            # {"primary_quantity":{"identifier":{"name": f"ion/{s}/density_fast",             "index":(idx:=idx+1) }}, "boundary_conditions": []},
            # {"primary_quantity":{"identifier":{"name": f"ion/{s}/temperature",              "index":(idx:=idx+1) }}, "boundary_conditions": []},
            # {"primary_quantity":{"identifier":{"name": f"ion/{s}/momentum",                 "index":(idx:=idx+1) }}, "boundary_conditions": []},
            ] for s,ion in  enumerate(core_profiles.time_slice.current.profiles_1d.ion)], [])
        ]
        # fmt:on

        self.time_slice.refresh(*args, {
            "primary_coordinate": self.code.parameters.primary_coordinate or "rho_tor_norm",
            "vacuum_toroidal_field": core_profiles.time_slice.current.vacuum_toroidal_field,
            "solver_1d": {
                "grid": core_profiles_1d.grid,
                "equation": equations
            }},
            core_profiles=core_profiles,
            equilibrium=equilibrium,
            core_transport=core_transport,
            core_sources=core_sources,
            ** kwargs)

        solver_1d: TransportSolverNumericsSolver1D = self.time_slice.current.solver_1d

        core_profiles_1d["grid"] = solver_1d.grid

        for eq in solver_1d.equation:
            core_profiles_1d[eq.primary_quantity.identifier.name] = eq.primary_quantity.profile

    def advance(self, *args, **kwargs) -> CoreProfiles.TimeSlice:
        self.time_slice.advance(*args, **kwargs)
