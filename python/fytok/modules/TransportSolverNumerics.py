from _imas.transport_solver_numerics import _T_transport_solver_numerics
from scipy import constants

from .CoreProfiles import CoreProfiles
from .CoreSources import CoreSources
from .CoreTransport import CoreTransport
from .Equilibrium import Equilibrium

from .Module import Module

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * constants.pi


class TransportSolverNumerics(_T_transport_solver_numerics, Module):
    r"""
        Solve transport equations
        :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`
    """

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
