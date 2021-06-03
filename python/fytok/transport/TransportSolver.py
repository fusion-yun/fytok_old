"""

"""

from dataclasses import dataclass
from math import log
from typing import Mapping, Optional

from fytok.common.Species import SpeciesIon
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


@dataclass
class _BC:
    identifier: Identifier
    value: np.ndarray
    rho_tor_norm: float


class TransportSolver(IDS):
    r"""
        Solve transport equations
        :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`
    """
    _IDS = "transport_solver_numerics"
    _actor_module_prefix = "transport.transport_solver."

    @dataclass
    class BoundaryConditions1D:

        @dataclass
        class Electrons:
            particles: _BC
            energy: _BC
            rho_tor_norm: float

        @dataclass
        class Ion(SpeciesIon):
            particles: _BC
            energy: _BC

        curent: _BC

        energy_ion_total: _BC

        momentum_tor: _BC

        electrons: Electrons

        ion: List[Ion]

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
    def boundary_condition_1d(self) -> BoundaryConditions1D:
        return self.get("boundary_condition_1d", {})

    def update(self, *args,  **kwargs):
        logger.debug(f"TODO: update dounbary condition")
        return 0.0

    def solve(self, /,
              core_profiles: CoreProfiles = None,
              equilibrium: Equilibrium = None,
              core_transport: CoreTransport = None,
              core_sources: CoreSources = None,
              **kwargs) -> float:
        """
            solve transport eqation
            return reduisal of core_profiles
        """
        logger.error(f"NOTIMPLEMENTED")
        return -1
