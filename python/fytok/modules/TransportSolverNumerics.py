from __future__ import annotations

from scipy import constants
from copy import copy
import math
from spdm.data.Expression import Expression, Variable, zero
from spdm.data.sp_property import sp_tree, sp_property, PropertyTree
from spdm.data.TimeSeries import TimeSlice, TimeSeriesAoS
from spdm.data.AoS import AoS
from spdm.utils.tags import _not_found_
from spdm.utils.typing import array_type

from .CoreProfiles import CoreProfiles
from .CoreSources import CoreSources
from .CoreTransport import CoreTransport
from .Equilibrium import Equilibrium
from .Utilities import *

from ..utils.logger import logger
from ..utils.atoms import atoms

# from ..ontology import transport_solver_numerics

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * constants.pi


@sp_tree
class TransportSolverNumericsEquation:
    """Profile and derivatives a the primary quantity for a 1D transport equation"""

    identifier: str
    """ Identifier of the primary quantity of the transport equation. The description
        node contains the path to the quantity in the physics IDS (example:
        core_profiles/profiles_1d/ion/D/density)"""

    profile: array_type
    """ Profile of the primary quantity"""

    flux: array_type
    """ Flux of the primary quantity"""

    coefficient: typing.List[typing.Any]
    """ Set of numerical coefficients involved in the transport equation
       
        [[u0,v0,v0],[u1,v1,w1],a,b,c,d,e,f,g,ym]
        a * y - b* ym + (1/c)* flux'= f - g*y
        flux=-d y' + e y
        u * y + v* flux - w =0 
    """

    boundary_condition_type: typing.Tuple[int, int]
    """ Identifier of the boundary condition type.  ID =
            1: value of the field y;
            2: radial derivative of the field (-dy/drho_tor);
            3: scale length of the field y/(-dy/drho_tor);
            4: flux;
            5: generic boundary condition y expressed as a1y'+a2y=a3.
            6: equation not solved;
    """
    boundary_value: typing.Tuple[typing.List[float], typing.List[float]]

    d_dr: array_type
    """ Radial derivative with respect to the primary coordinate"""

    dflux_dr: array_type
    """ Radial derivative of Flux of the primary quantity"""

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

    convergence: PropertyTree
    """ Convergence details"""


@sp_tree(coordinate1="grid/rho_tor_norm")
class TransportSolverNumericsTimeSlice(TimeSlice):
    """Numerics related to 1D radial solver for a given time slice"""

    grid: CoreRadialGrid
    """ Radial grid"""

    primary_coordinate: Variable

    variables: typing.OrderedDict[str, Expression | array_type]

    equations: AoS[TransportSolverNumericsEquation]
    """ Set of transport equations"""

    control_parameters: PropertyTree
    """ Solver-specific input or output quantities"""

    drho_tor_dt: Expression = sp_property(units="m.s^-1")
    """ Partial derivative of the toroidal flux coordinate profile with respect to time"""

    d_dvolume_drho_tor_dt: Expression = sp_property(units="m^2.s^-1")
    """ Partial derivative with respect to time of the derivative of the volume with
      respect to the toroidal flux coordinate"""


@sp_tree
class TransportSolverNumerics(IDS):
    r"""Solve transport equations  $\rho=\sqrt{ \Phi/\pi B_{0}}$"""

    _plugin_prefix = "fytok.plugins.transport_solver_numerics."

    code: Code = {"name": "fy_trans"}

    solver: str = "ion_solver"

    primary_coordinate: str | Variable = "rho_tor_norm"
    r""" 与 core_profiles 的 primary coordinate 磁面坐标一致
      rho_tor_norm $\bar{\rho}_{tor}=\sqrt{ \Phi/\Phi_{boundary}}$ """

    species: typing.List[str]

    boundary_condition_type: typing.Dict[str, typing.Any]

    variables: typing.Dict[str, Expression]

    equations: AoS[TransportSolverNumericsEquation]

    TimeSlice = TransportSolverNumericsTimeSlice

    time_slice: TimeSeriesAoS[TransportSolverNumericsTimeSlice]

    def __init__(self, *args, **kwargs):
        prev_cls = self.__class__
        super().__init__(*args, **kwargs)
        if self.__class__ is not prev_cls:
            return

        if self.solver != "ion_solver":
            raise NotImplementedError(f"Only ion_solver is supported!")

        enabla_momentum = self.code.parameters.enabla_momentum

        enabla_current_diffusion = self.code.parameters.enabla_current_diffusion

        equations = []

        # 定义 equation 和 variable
        if self.primary_coordinate == "rho_tor_norm":
            self._cache["primary_coordinate"] = Variable(
                (index := 0), self.primary_coordinate, label=r"\bar{\rho}_{tor}"
            )

        elif isinstance(self.primary_coordinate, str):
            self._cache["primary_coordinate"] = Variable((index := 0), self.primary_coordinate)

        else:
            index = 0

        variables = {}

        if enabla_current_diffusion:
            variables["psi"] = Variable(
                (index := index + 1),
                "psi",
                label=r"\psi",
            )
            variables["psi_flux"] = Variable(
                (index := index + 1),
                "psi",
                label=r"\Gamma_{\psi}",
            )

            equations.append(
                {
                    "identifier": "psi",
                    "boundary_condition_type": self.boundary_condition_type.get("psi", (2, 1)),
                }
            )

        variables["electrons/temperature"] = Variable(
            (index := index + 1),
            "electrons/temperature",
            label=r"T_{e}",
        )

        variables["electrons/temperature_flux"] = Variable(
            (index := index + 1),
            "electrons/temperature",
            label=r"H_{Te}",
        )

        equations.append(
            {
                "identifier": "electrons/temperature",
                "boundary_condition_type": self.boundary_condition_type.get("electrons/temperature", (2, 1)),
            }
        )

        n_e = zero
        flux_e = zero
        for s in self.species:
            variables[f"ion/{s}/density"] = ns = Variable((index := index + 1), f"ion/{s}/density", label=rf"n_{s}")

            variables[f"ion/{s}/density_flux"] = ns_flux = Variable(
                (index := index + 1), f"ion/{s}/density_flux", label=rf"\Gamma_{s}"
            )
            equations.append(
                {
                    "identifier": f"ion/{s}/density",
                    "boundary_condition_type": self.boundary_condition_type.get(f"ion/{s}/density", (2, 1)),
                }
            )

            variables[f"ion/{s}/temperature"] = Variable(
                (index := index + 1), f"ion/{s}/temperature", label=rf"T_{{{s}}}"
            )

            variables[f"ion/{s}/temperature_flux"] = Variable(
                (index := index + 1), f"ion/{s}/temperature_flux", label=rf"H_{{{s}}}"
            )

            equations.append(
                {
                    "identifier": f"ion/{s}/temperature",
                    "boundary_condition_type": self.boundary_condition_type.get(f"ion/{s}/temperature", (2, 1)),
                }
            )

            if enabla_momentum:
                variables[f"ion/{s}/momentum"] = ns = Variable(
                    (index := index + 1), f"ion/{s}/momentum", label=rf"u_{s}"
                )
                variables[f"ion/{s}/momentum_flux"] = ns_flux = Variable(
                    (index := index + 1), f"ion/{s}/momentum_flux", label=rf"U_{s}"
                )

                equations.append(
                    {
                        "identifier": f"ion/{s}/momentum",
                        "boundary_condition_type": self.boundary_condition_type.get(f"ion/{s}/momentum", (2, 1)),
                    }
                )

            z = atoms[s].z
            n_e -= z * ns
            flux_e -= z * ns_flux

        # quasi neutrality condition
        variables["electrons/density"] = n_e
        variables["electrons/density_flux"] = flux_e

        self._cache["variables"] = variables
        self["equations"] = equations

        logger.debug(f" Variables : {[k for k,v in self.variables.items() if isinstance(v,Variable)]}")

    def preprocess(
        self, *args, grid=_not_found_, initial_value=None, boundary_value=None, **kwargs
    ) -> TransportSolverNumericsTimeSlice:
        current = super().preprocess(*args, **kwargs)

        equilibrium: Equilibrium.TimeSlice = self.inputs.get_source("equilibrium").time_slice.current

        assert math.isclose(equilibrium.time, self.time), f"{equilibrium.time} != {current.time}"

        # TODO: 根据时间获取时间片, 例如：
        #   eq:Equilibrium.TimeSlice= equilibrium.time_slice.get(self.time)
        #   current["grid"] = eq.profiles_1d.grid.duplicate(rho_tor_norm)

        if grid is _not_found_:
            rho_tor_norm = self.code.parameters.get("rho_tor_norm", None)
            current["grid"] = equilibrium.profiles_1d.grid.remesh(rho_tor_norm)
        else:
            current["grid"] = equilibrium.profiles_1d.grid.remesh(grid)

        if initial_value is None:
            initial_value = {}
        if boundary_value is None:
            boundary_value = {}

        core_profiles: CoreProfiles = self.inputs.get_source("core_profiles")
        core_profiles_1d = core_profiles.time_slice.current.profiles_1d

        current["equations"] = [
            {
                "identifier": equ.identifier,
                "profile": initial_value.get(equ.identifier, zero),
                "flux": initial_value.get(f"{equ.identifier}_flux", zero),
                "boundary_condition_type": equ.boundary_condition_type,
                "boundary_value": boundary_value.get(equ.identifier, [0, 0]),
            }
            for idx, equ in enumerate(self.equations)
        ]

        current["control_parameters"] = self.code.parameters

        return current

    def execute(
        self, current: TransportSolverNumericsTimeSlice, *previous: TransportSolverNumericsTimeSlice
    ) -> TransportSolverNumericsTimeSlice:
        logger.info(f"Solve transport equations : { '  ,'.join([equ.identifier for equ in self.equations])}")
        return super().execute(current, *previous)

    def refresh(
        self,
        *args,
        equilibrium: Equilibrium = None,
        core_transport: CoreTransport = None,
        core_sources: CoreSources = None,
        core_profiles: CoreProfiles = None,
        **kwargs,
    ) -> TransportSolverNumericsTimeSlice:
        return super().refresh(
            *args,
            equilibrium=equilibrium,
            core_transport=core_transport,
            core_sources=core_sources,
            core_profiles=core_profiles,
            **kwargs,
        )
