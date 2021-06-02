"""

"""

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


class TransportSolverBoundaryConditions1DBC(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def identifier(self) -> Identifier:
        return self["identifier"]

    @sp_property
    def value(self) -> np.ndarray:
        return self["value"]

    @sp_property
    def rho_tor_norm(self) -> float:
        return self["rho_tor_norm"] or 0.99


class TransportSolverBoundaryConditions1DBCElectrons(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def particles(self) -> TransportSolverBoundaryConditions1DBC:
        return self["particles"]

    @sp_property
    def energy(self) -> TransportSolverBoundaryConditions1DBC:
        return self["value"]

    @sp_property
    def rho_tor_norm(self) -> float:
        return self["rho_tor_norm"] or 0.99


class TransportSolverBoundaryConditions1DBCIon(SpeciesIon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def particles(self) -> TransportSolverBoundaryConditions1DBC:
        return self["particles"]

    @sp_property
    def energy(self) -> TransportSolverBoundaryConditions1DBC:
        return self["value"]


class TransportSolverBoundaryConditions1D(AttributeTree):
    BC = TransportSolverBoundaryConditions1DBC
    BCElectrons = TransportSolverBoundaryConditions1DBCElectrons
    BCIon = TransportSolverBoundaryConditions1DBCIon

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def curent(self) -> BC:
        return self["current"]

    @sp_property
    def energy_ion_total(self) -> BC:
        return self["energy_ion_total"]

    @sp_property
    def momentum_tor(self) -> BC:
        return self["momentum_tor"]

    @sp_property
    def electrons(self) -> BCElectrons:
        return self["electrons"]

    @sp_property
    def ion(self) -> List[BCIon]:
        return self["ion"]


class TransportSolver(IDS):
    r"""
        Solve transport equations
        :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`
    """
    _IDS = "transport_solver_numerics"
    BoundaryConditions1D = TransportSolverBoundaryConditions1D
    _actor_module_prefix = "transport.transport_solver."

    def __init__(self,  *args, grid: RadialGrid = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._grid = grid if grid is not None else self._parent._grid

    @property
    def grid(self):
        return self._grid

    @sp_property
    def solver(self) -> Identifier:
        return self["solver"]

    @sp_property
    def primary_coordinate(self) -> Identifier:
        return self["primary_coordinate"]

    @sp_property
    def boundary_condition_1d(self) -> BoundaryConditions1D:
        return self["boundary_condition_1d"]

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
