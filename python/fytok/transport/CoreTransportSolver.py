"""

"""

from dataclasses import dataclass, field
from math import log
from typing import Mapping, Optional, Tuple

from fytok.common.Species import SpeciesElectron, SpeciesIon
from spdm.data.Function import Function
from spdm.data.Node import Dict, List, _not_found_, sp_property
from spdm.numlib import constants, np
from spdm.util.logger import logger

from ..common.IDS import IDS
from ..common.Misc import Identifier
from ..common.Atoms import atoms

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


class CoreTransportSolver(IDS):
    r"""
        Solve transport equations
        :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`
    """
    _IDS = "transport_solver_numerics"
    _actor_module_prefix = "fymodules.transport.core_transport_solver."

    class BoundaryConditions1D(Dict):
        BoundaryConditions = _BC

        def __init__(self, *args, grid: RadialGrid = None,  **kwargs):
            super().__init__(*args,  ** kwargs)
            self._grid = grid if grid is not None else getattr(self._parent, "_grid", None)

        @property
        def grid(self) -> RadialGrid:
            return self._grid

        class Electrons(SpeciesElectron):
            def __init__(self, *args, **kwargs):
                super().__init__(*args,   **kwargs)

            @sp_property
            def particles(self) -> _BC:
                return self.get("particles")

            @sp_property
            def energy(self) -> _BC:
                return self.get("energy")

            @sp_property
            def rho_tor_norm(self) -> float:
                return self.get("rho_tor_norm", 1.0)

        class Ion(SpeciesIon):
            def __init__(self, *args, **kwargs):
                super().__init__(*args,   **kwargs)

            @sp_property
            def particles(self) -> _BC:
                return self.get("particles")

            @sp_property
            def energy(self) -> _BC:
                return self.get("energy")

            @sp_property
            def rho_tor_norm(self) -> float:
                return self.get("rho_tor_norm", 1.0)

        @sp_property
        def electrons(self) -> Electrons:
            return CoreTransportSolver.BoundaryConditions1D.Electrons(self.get("electrons"), parent=self, grid=self._grid)

        @sp_property
        def ion(self) -> List[Ion]:
            return self.get("ion")

        @sp_property
        def current(self) -> BoundaryConditions:
            return self.get("current")

        @sp_property
        def energy_ion_total(self) -> BoundaryConditions:
            return self.get("energy_ion_total")

        @sp_property
        def momentum_tor(self) -> BoundaryConditions:
            return self.get("momentum_tor")

    def __init__(self, *args, **kwargs):
        super().__init__(*args,  ** kwargs)

    @sp_property
    def solver(self) -> Identifier:
        return self.get("solver")

    @sp_property
    def primary_coordinate(self) -> Identifier:
        return self.get("primary_coordinate")

    @sp_property
    def boundary_conditions_1d(self) -> BoundaryConditions1D:
        return self.get("boundary_conditions_1d")

    def refresh(self, *args,  boundary_conditions_1d=None,  **kwargs):
        if boundary_conditions_1d is not None:
            self.boundary_conditions_1d.update(boundary_conditions_1d)
        return 0.0

    def solve(self, /,
              core_profiles_next: CoreProfiles,
              core_profiles_prev: CoreProfiles,
              core_transport: CoreTransport.Model,
              core_sources: CoreSources.Source,
              equilibrium_next: Equilibrium,
              equilibrium_prev: Equilibrium = None,
              dt: float = None,
              **kwargs) -> float:
        """
            solve transport eqation until residual < tolerance
            return residual , core_profiles, edge_profiles
        """

        profiles = core_profiles_next.profiles_1d
        profiles["grid"] = core_profiles_prev.profiles_1d.grid

        psi_norm = profiles.grid.psi_norm
        rho_tor_norm = profiles.grid.rho_tor_norm

        # profiles["q"] = equilibrium_1d.q(psi_norm)
        # profiles["magnetic_shear"] = equilibrium_1d.magnetic_shear(psi_norm)

        # profiles["conductivity_parallel"] = core_transport_1d.conductivity_parallel(rho_tor_norm)
        # profiles["j_tor"] = equilibrium_1d.j_tor(psi_norm)
        # profiles["j_total"] = core_sources_1d.j_parallel(rho_tor_norm)
        # profiles["j_bootstrap"] = core_transport_1d.fetch('j_bootstrap')(rho_tor_norm)

        profiles["electrons"] = {**atoms["e"]}
        profiles["ion"] = [
            {**atoms[ion.label],
             "z_ion_1d":ion.z_ion_1d(rho_tor_norm),
             "is_impurity":ion.is_impurity,
             "density":ion.density(rho_tor_norm),
             "energy":ion.temperature(rho_tor_norm)}
            for ion in core_profiles_prev.profiles_1d.ion
        ]

        return 0.0
