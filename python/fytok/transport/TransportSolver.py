"""

"""

from dataclasses import dataclass, field
from math import log
from typing import Mapping, Optional

from fytok.common.Species import SpeciesElectron, SpeciesIon
from fytok.transport.EdgeProfiles import EdgeProfiles
from fytok.transport.EdgeSources import EdgeSources
from fytok.transport.EdgeTransport import EdgeTransport
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Node import Dict, List, _not_found_, sp_property
from spdm.numlib import constants, np
from spdm.util.logger import logger

from ..common.IDS import IDS
from ..common.Misc import Identifier
from .CoreProfiles import CoreProfiles
from .CoreSources import CoreSources
from .CoreTransport import CoreTransport
from .Equilibrium import Equilibrium
from .MagneticCoordSystem import RadialGrid

# from .EdgeProfiles import EdgeProfiles
# from .EdgeSources import EdgeSources
# from .EdgeTransport import EdgeTransport
EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * constants.pi


class _BC(Dict):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def value(self) -> np.ndarray:
        return self.get('value', [0.0])

    @sp_property
    def rho_tor_norm(self) -> float:
        return self.get('rho_tor_norm', 1.0)

    @sp_property
    def identifier(self) -> Identifier:
        return self.get('identifier', {"index": 1})


class TransportSolver(IDS):
    r"""
        Solve transport equations
        :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`
    """
    _IDS = "transport_solver_numerics"
    _actor_module_prefix = "transport.transport_solver."

    class BoundaryConditions1D(Dict):
        BoundaryConditions = _BC

        def __init__(self,   *args, grid: RadialGrid = None, **kwargs):
            super().__init__(*args, **kwargs)
            self._grid = grid or self._parent._grid

        @property
        def grid(self) -> RadialGrid:
            return self._grid

        class Electrons(SpeciesElectron):
            def __init__(self, *args, **kwargs):
                super().__init__(*args,   **kwargs)

            @sp_property
            def particles(self) -> _BC:
                return self.get("particles", {})

            @sp_property
            def energy(self) -> _BC:
                return self.get("energy", {})

            @sp_property
            def rho_tor_norm(self) -> float:
                return self.get("rho_tor_norm", 1.0)

        class Ion(SpeciesIon):
            def __init__(self, *args, **kwargs):
                super().__init__(*args,   **kwargs)

            @sp_property
            def particles(self) -> _BC:
                return self.get("particles", {})

            @sp_property
            def energy(self) -> _BC:
                return self.get("energy", {})

            @sp_property
            def rho_tor_norm(self) -> float:
                return self.get("rho_tor_norm", 1.0)

        @sp_property
        def electrons(self) -> Electrons:
            return self.get("electrons", {})

        @sp_property
        def current(self) -> BoundaryConditions:
            return self.get("current", {})

        @sp_property
        def energy_ion_total(self) -> BoundaryConditions:
            return self.get("energy_ion_total", {})

        @sp_property
        def momentum_tor(self) -> BoundaryConditions:
            return self.get("momentum_tor", {})

        @sp_property
        def ion(self) -> List[Ion]:
            return self.get("ion", [])

    def __init__(self,  *args, grid: RadialGrid = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._grid = grid if grid is not None else self._parent._grid

    @property
    def grid(self):
        return self._grid

    @sp_property
    def solver(self) -> Identifier:
        return self.get("solver", {})

    @sp_property
    def primary_coordinate(self) -> Identifier:
        return self.get("primary_coordinate", {})

    @sp_property
    def boundary_conditions_1d(self) -> BoundaryConditions1D:
        return self.get("boundary_conditions_1d", {})

    def update(self, /,
               equilibrium: Equilibrium = None,
               core_profiles: CoreProfiles = None,
               core_transport: CoreTransport = None,
               core_sources: CoreSources = None,
               edge_profiles: EdgeProfiles = False,
               edge_transport: EdgeTransport = False,
               edge_sources: EdgeSources = False,
               **kwargs):

        if equilibrium is not None:
            self._equilibrium = equilibrium
        if core_profiles is not None:
            self._core_profiles = core_profiles
        if core_transport is not None:
            self._core_transport = core_transport
        if core_sources is not None:
            self._core_sources = core_sources
        if edge_profiles is not None:
            self._edge_profiles = edge_profiles
        if edge_transport is not None:
            self._edge_transport = edge_transport
        if edge_sources is not None:
            self._edge_sources = edge_sources

        return 0.0

    def solve_core(self, *args, max_nodes=1000, tolerance=1e-3, **kwargs):
        return NotImplemented

    def solve_core(self, *args,   tolerance=1e-3, **kwargs):
        return NotImplemented

    def solve(self,  max_iter=1, tolerance=1.0e-3, ** kwargs) -> float:
        """
            solve transport eqation
            return residual of core_profiles
        """

        for step in range(max_iter):
            logger.debug(f" Iteration step={step}: start")

            residual = []
            residual.append(self.solve_core(tolerance=tolerance, **kwargs))

            if self._edge_profiles is not False:
                residual.append(self.solve_edge(tolerance=tolerance, **kwargs))

            residual = max(residual)

            logger.debug(f" Iteration step={step}: stop   residual={residual}  ")

            if residual < tolerance:
                break

        return residual
