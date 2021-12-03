"""

"""

from dataclasses import dataclass, field
from math import log
from typing import Mapping, Optional, Tuple

import numpy as np
from scipy import constants
from spdm.common.logger import logger
from spdm.data.Dict import Dict
from spdm.data.Entry import Entry
from spdm.data.Function import Function, function_like
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property

from ..common.Atoms import atoms
from ..common.IDS import IDS
from ..common.Misc import Identifier
from ..common.Species import SpeciesElectron, SpeciesIon
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
    _fy_module_prefix = "fymodules.transport.core_transport_solver."

    class BoundaryConditions1D(Dict):
        BoundaryConditions = _BC

        def __init__(self, *args,   **kwargs):
            super().__init__(*args,  ** kwargs)

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
            def particles_fast(self) -> _BC:
                return self.get("particles_fast")

            @sp_property
            def energy(self) -> _BC:
                return self.get("energy")

            @sp_property
            def rho_tor_norm(self) -> float:
                return self.get("rho_tor_norm", 1.0)

        @sp_property
        def electrons(self) -> Electrons:
            return self.get("electrons", {})

        @sp_property
        def ion(self) -> List[Ion]:
            return self.get("ion", [])

        @sp_property
        def current(self) -> BoundaryConditions:
            return self.get("current", {})

        @sp_property
        def energy_ion_total(self) -> BoundaryConditions:
            return self.get("energy_ion_total", {})

        @sp_property
        def momentum_tor(self) -> BoundaryConditions:
            return self.get("momentum_tor", {})

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
        return self.get("boundary_conditions_1d", {})

    def refresh(self, *args,  boundary_conditions_1d=None,  **kwargs):
        if boundary_conditions_1d is not None:
            self.boundary_conditions_1d.update(boundary_conditions_1d)
        return 0.0

    def solve(self, /,
              core_profiles_prev: CoreProfiles,
              core_transport: CoreTransport,
              core_sources: CoreSources,
              equilibrium_next: Equilibrium,
              equilibrium_prev: Equilibrium = None,
              dt: float = None,
              **kwargs) -> CoreProfiles:
        """
            solve transport equation until residual < tolerance
            return core_profiles
        """
        raise NotImplementedError()

        # return CoreProfiles({
        #     "profiles_1d": {
        #         "grid": core_profiles_prev.profiles_1d.grid,
        #         "electrons": {"label": "e"},
        #         "ion": [{"label": ion.label} for ion in core_profiles_prev.profiles_1d.ion]
        #     }
        # })
