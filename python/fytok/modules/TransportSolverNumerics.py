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
    boundary_value: tuple  # typing.Tuple[typing.List[float], typing.List[float]]

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

    variables: typing.Dict[str, Expression]

    equations: AoS[TransportSolverNumericsEquation]
    """ Set of transport equations"""

    control_parameters: PropertyTree
    """ Solver-specific input or output quantities"""

    drho_tor_dt: Expression = sp_property(units="m.s^-1")
    """ Partial derivative of the toroidal flux coordinate profile with respect to time"""

    d_dvolume_drho_tor_dt: Expression = sp_property(units="m^2.s^-1")
    """ Partial derivative with respect to time of the derivative of the volume with
      respect to the toroidal flux coordinate"""

    Y0: array_type = None


@sp_tree
class TransportSolverNumerics(IDS):
    r"""Solve transport equations  $\rho=\sqrt{ \Phi/\pi B_{0}}$"""

    _plugin_prefix = "fytok.plugins.transport_solver_numerics."

    code: Code = {"name": "fy_trans"}

    solver: str = "ion_solver"

    primary_coordinate: str = "rho_tor_norm"
    r""" 与 core_profiles 的 primary coordinate 磁面坐标一致
      rho_tor_norm $\bar{\rho}_{tor}=\sqrt{ \Phi/\Phi_{boundary}}$ """

    thermal_particle: typing.List[str] = []
    fast_particle: typing.List[str] = []
    impurities: typing.List[str] = []

    boundary_condition_type: typing.Dict[str, typing.Any]

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

    def preprocess(self, *args, initial_value=None, boundary_value=None, **kwargs) -> TransportSolverNumericsTimeSlice:
        current: TransportSolverNumericsTimeSlice = super().preprocess(*args, **kwargs)
        core_profiles: CoreProfiles = self.inputs.get_source("core_profiles")
        core_profiles_1d = core_profiles.time_slice.current.profiles_1d

        grid = current.cache_get("grid", _not_found_)

        if not isinstance(grid, CoreRadialGrid):
            equilibrium: Equilibrium.TimeSlice = self.inputs.get_source("equilibrium").time_slice.current

            # TODO: 根据时间获取时间片, 例如：
            # assert math.isclose(equilibrium.time, self.time), f"{equilibrium.time} != {current.time}"
            #   eq:Equilibrium.TimeSlice= equilibrium.time_slice.get(self.time)
            #   current["grid"] = eq.profiles_1d.grid.duplicate(rho_tor_norm)

            rho_tor_norm = kwargs.get("rho_tor_norm", self.code.parameters.get("rho_tor_norm", None))

            current.grid = equilibrium.profiles_1d.grid.remesh(rho_tor_norm)

        if initial_value is None:
            initial_value = {}
        if boundary_value is None:
            boundary_value = {}

        enabla_momentum = self.code.parameters.enabla_momentum

        enabla_current_diffusion = self.code.parameters.enabla_current_diffusion

        equations: typing.List[typing.Dict[str, typing.Any]] = []

        # 定义 equation 和 variable
        if self.primary_coordinate == "rho_tor_norm":
            x = Variable((i := 0), self.primary_coordinate, label=r"\bar{\rho}_{tor}")

        else:
            x = Variable((i := 0), self.primary_coordinate)

        current["primary_coordinate"] = x

        variables = {}

        bc_type = self.boundary_condition_type

        if enabla_current_diffusion:
            variables["psi"] = Variable((i := i + 1), "psi", label=r"\psi")
            variables["psi_flux"] = Variable((i := i + 1), "psi", label=r"\Gamma_{\psi}")
            equations.append({"identifier": "psi", "boundary_condition_type": bc_type.get("psi", (2, 1))})

        n_e = zero

        flux_e = zero

        for s in self.thermal_particle:
            variables[f"ion/{s}/density"] = ns = Variable(
                (i := i + 1),
                f"ion/{s}/density",
                label=rf"n_{s}",
            )

            variables[f"ion/{s}/density_flux"] = ns_flux = Variable(
                (i := i + 1),
                f"ion/{s}/density_flux",
                label=rf"\Gamma_{s}",
            )

            equations.append(
                {
                    "identifier": f"ion/{s}/density",
                    "boundary_condition_type": bc_type.get(f"ion/{s}/density", None)
                    or bc_type.get(f"*/density", (2, 1)),
                }
            )

            variables[f"ion/{s}/temperature"] = Variable(
                (i := i + 1),
                f"ion/{s}/temperature",
                label=rf"T_{{{s}}}",
            )

            variables[f"ion/{s}/temperature_flux"] = Variable(
                (i := i + 1),
                f"ion/{s}/temperature_flux",
                label=rf"H_{{{s}}}",
            )

            equations.append(
                {
                    "identifier": f"ion/{s}/temperature",
                    "boundary_condition_type": bc_type.get(f"ion/{s}/temperature", None)
                    or bc_type.get(f"*/temperature", (2, 1)),
                }
            )

            if enabla_momentum:
                variables[f"ion/{s}/momentum"] = ns = Variable(
                    (i := i + 1),
                    f"ion/{s}/momentum",
                    label=rf"u_{s}",
                )

                variables[f"ion/{s}/momentum_flux"] = ns_flux = Variable(
                    (i := i + 1),
                    f"ion/{s}/momentum_flux",
                    label=rf"U_{s}",
                )

                equations.append(
                    {
                        "identifier": f"ion/{s}/momentum",
                        "boundary_condition_type": bc_type.get(f"ion/{s}/momentum", None)
                        or bc_type.get(f"*/momentum", (2, 1)),
                    }
                )

            z = atoms[s].z
            n_e += z * ns
            flux_e += z * ns_flux

        for s in self.fast_particle:
            variables[f"ion/{s}/density"] = ns = Variable(
                (i := i + 1),
                f"ion/{s}/density",
                label=rf"n_{s}",
            )

            variables[f"ion/{s}/density_flux"] = ns_flux = Variable(
                (i := i + 1),
                f"ion/{s}/density_flux",
                label=rf"\Gamma_{s}",
            )

            equations.append(
                {
                    "identifier": f"ion/{s}/density",
                    "boundary_condition_type": bc_type.get(f"ion/{s}/density", None)
                    or bc_type.get(f"*/density", (2, 1)),
                }
            )

        variables["electrons/temperature"] = Variable(
            (i := i + 1),
            "electrons/temperature",
            label=r"T_{e}",
        )

        variables["electrons/temperature_flux"] = Variable(
            (i := i + 1),
            "electrons/temperature",
            label=r"H_{Te}",
        )

        equations.append(
            {
                "identifier": "electrons/temperature",
                "boundary_condition_type": bc_type.get("electrons/temperature", None)
                or bc_type.get(f"*/temperature", (2, 1)),
            }
        )

        for s in self.impurities:
            z_ion_1d = core_profiles_1d.get(f"ion/{s}/z_ion_1d", 0)
            n_e += (core_profiles_1d.get(f"ion/{s}/density", 0) * z_ion_1d)(x)
            flux_e += (core_profiles_1d.get(f"ion/{s}/density_flux", 0) * z_ion_1d)(x)

        # quasi neutrality condition
        variables["electrons/density"] = n_e
        variables["electrons/density_flux"] = flux_e

        current["equations"] = equations

        current["control_parameters"] = self.code.parameters

        current._cache["variables"] = variables

        for equ in current.equations:
            equ["boundary_value"] = boundary_value.get(equ.identifier, None)
            equ["profile"] = initial_value.get(equ.identifier, zero)
            equ["flux"] = initial_value.get(f"{equ.identifier}_flux", zero)

        logger.debug(f" Variables : {[k for k,v in current.variables.items() if isinstance(v,Variable)]}")

        return current

    def execute(
        self,
        current: TransportSolverNumericsTimeSlice,
        *previous: TransportSolverNumericsTimeSlice,
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

    def fetch(self, *args, **kwargs) -> CoreProfiles.TimeSlice.Profiles1D:
        """获得 CoreProfiles.TimeSlice.Profiles1D 形式状态树。"""
        current = self.time_slice.current

        data = {
            "grid": current.grid,
            "ion": [{"label": s} for s in self.thermal_particle],
        }

        X = current.grid.rho_tor_norm
        Y = current.Y0
        for k, v in current.variables.items():
            pth = Path(k)
            if pth[0] == "ion":
                spec = pth[1]
                try:
                    idx = self.thermal_particle.index(spec)
                except Exception:
                    logger.warning(f"ignore {k}")
                    continue
                else:
                    pth[1] = idx

            data = pth.update(data, v(X, *Y))

        return CoreProfiles.TimeSlice.Profiles1D(data)
