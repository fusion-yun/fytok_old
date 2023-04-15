"""

"""


import numpy as np
from scipy import constants
from spdm.data.Dict import Dict
from spdm.data.List import List
from spdm.data.sp_property import sp_property

from ..common.IDS import IDS
from ..common.Misc import Identifier
from ..transport.CoreProfiles import CoreProfiles
from ..transport.CoreSources import CoreSources
from ..transport.CoreTransport import CoreTransport
from ..transport.Equilibrium import Equilibrium
from .Species import SpeciesElectron, SpeciesIon

# from .EdgeProfiles import EdgeProfiles
# from .EdgeSources import EdgeSources
# from .EdgeTransport import EdgeTransport
EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * constants.pi

class _BC(Dict):

    value: np.ndarray = sp_property()

    rho_tor_norm: float = sp_property()

    identifier: Identifier = sp_property()


class CoreTransportSolver(IDS):
    r"""
        Solve transport equations
        :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`
    """
    _IDS = "core_transport_solver"

    class BoundaryConditions1D(Dict):
        BoundaryConditions = _BC

        class Electrons(SpeciesElectron):
            particles: _BC = sp_property()

            energy: _BC = sp_property()

            rho_tor_norm: float = sp_property(default=1.0)

        class Ion(SpeciesIon):
            particles: _BC = sp_property()

            particles_fast: _BC = sp_property()

            energy: _BC = sp_property()

            rho_tor_norm: float = sp_property(default=1.0)

        electrons: Electrons = sp_property()

        ion: List[Ion] = sp_property()

        current: BoundaryConditions = sp_property()

        energy_ion_total: BoundaryConditions = sp_property()

        momentum_tor: BoundaryConditions = sp_property()

    solver: Identifier = sp_property()

    primary_coordinate: Identifier = sp_property()

    boundary_conditions_1d: BoundaryConditions1D = sp_property()

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
