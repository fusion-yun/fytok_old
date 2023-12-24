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

    profile: array_type | Expression
    """ Profile of the primary quantity"""

    flux: array_type | Expression
    """ Flux of the primary quantity"""

    d_dr: array_type | Expression
    """ Radial derivative with respect to the primary coordinate"""

    dflux_dr: array_type | Expression
    """ Radial derivative of Flux of the primary quantity"""

    d2_dr2: array_type | Expression
    """ Second order radial derivative with respect to the primary coordinate"""

    d_dt: array_type | Expression
    """ Time derivative"""

    d_dt_cphi: array_type | Expression
    """ Derivative with respect to time, at constant toroidal flux (for current
        diffusion equation)"""

    d_dt_cr: array_type | Expression
    """ Derivative with respect to time, at constant primary coordinate coordinate (for
        current diffusion equation)"""

    coefficient: typing.List[typing.Any]
    """ Set of numerical coefficients involved in the transport equation
       
        [d_dt,D,V,RHS]
        
        d_dt + flux'= RHS  
        
        flux =-D y' + V y

        u * y + v* flux - w =0 
    """

    boundary_condition_type: int = 1

    boundary_condition_value: tuple
    """ [u,v,v] 
    
    u * profile + v* flux - w =0"""

    convergence: PropertyTree
    """ Convergence details"""


@sp_tree(coordinate1="grid/rho_tor_norm")
class TransportSolverNumericsTimeSlice(TimeSlice):
    """Numerics related to 1D radial solver for a given time slice"""

    grid: CoreRadialGrid
    """ Radial grid"""

    primary_coordinate: str

    equations: AoS[TransportSolverNumericsEquation]
    """ Set of transport equations"""

    control_parameters: PropertyTree
    """ Solver-specific input or output quantities"""

    drho_tor_dt: array_type | Expression = sp_property(units="m.s^-1")
    """ Partial derivative of the toroidal flux coordinate profile with respect to time"""

    d_dvolume_drho_tor_dt: array_type | Expression = sp_property(units="m^2.s^-1")
    """ Partial derivative with respect to time of the derivative of the volume with
      respect to the toroidal flux coordinate"""


@sp_tree
class TransportSolverNumerics(IDS):
    r"""Solve transport equations  $\rho=\sqrt{ \Phi/\pi B_{0}}$"""

    _plugin_prefix = "fytok.plugins.transport_solver_numerics."

    code: Code = {"name": "fy_trans"}

    solver: str = "ion_solver"

    fusion_reactions: set

    ion: set = set()

    neutral: set

    impurities: set

    primary_coordinate: str | Variable = "rho_tor_norm"
    r""" 与 core_profiles 的 primary coordinate 磁面坐标一致
      rho_tor_norm $\bar{\rho}_{tor}=\sqrt{ \Phi/\Phi_{boundary}}$ """

    equations: AoS[TransportSolverNumericsEquation]

    variables: typing.Dict[str, Expression]

    TimeSlice = TransportSolverNumericsTimeSlice

    time_slice: TimeSeriesAoS[TransportSolverNumericsTimeSlice]

    def __init__(self, *args, **kwargs):
        prev_cls = self.__class__
        super().__init__(*args, **kwargs)
        if self.__class__ is not prev_cls:
            return

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
        current: TransportSolverNumericsTimeSlice = super().fetch()
        X = current.grid.rho_tor_norm
        Y = sum([[equ.profile, equ.flux] for equ in current.equations], [])

        profiles_1d = CoreProfiles.TimeSlice.Profiles1D(
            {
                "grid": current.grid,
                "ion": [
                    {"label": equ.identifier.split("/")[-2]}
                    for equ in current.equations
                    if equ.identifier.endswith("/density") and equ.identifier.startswith("ion/")
                ],
            }
        )
        for k, v in self.variables.items():
            profiles_1d[k] = v(X, *Y)

        return profiles_1d
