from scipy import constants
from fytok.utils.logger import logger
from spdm.data.sp_property import sp_tree
from .CoreProfiles import CoreProfiles
from .CoreSources import CoreSources
from .CoreTransport import CoreTransport
from .Equilibrium import Equilibrium

from ..ontology import transport_solver_numerics

EPSILON = 1.0e-15
TOLERANCE = 1.0e-6

TWOPI = 2.0 * constants.pi


class TransportSolverNumerics(transport_solver_numerics._T_transport_solver_numerics):
    r"""
        Solve transport equations
        :math:`\rho=\sqrt{ \Phi/\pi B_{0}}`
    """
    _plugin_prefix = 'fytok.plugins.transport_solver_numerics.'

    def solve_15D_adv(self, *args, tolerance=1.0e-4, max_iteration=1, **kwargs):
        self._time += dt

        core_profiles_1d_prev = self.core_profiles.profiles_1d.current

        equilibrium = self.equilibrium.advance(
            time=self.time,
            core_profile_1d=core_profiles_1d_prev,
            wall=self.wall,
            pf_active=self.pf_active,
        )

        core_transport_profiles_1d = self.core_transport.advance(
            time=self.time,
            equilibrium=equilibrium,
            core_profile_1d=core_profiles_1d_prev,
        )

        core_source_profiles_1d = self.core_sources.advance(
            time=self.time,
            equilibrium=equilibrium,
            core_profile_1d=core_profiles_1d_prev,
        )

        core_profiles_1d_next = self.transport_solver.solve(
            equilibrium=equilibrium,
            core_profile_1d=core_profiles_1d_prev,
            core_transport_profiles_1d=core_transport_profiles_1d,
            core_source_profiles_1d=core_source_profiles_1d,
        )

        self.core_profiles.advance(core_profiles_1d_next)

        self.core_profiles.advance(core_profiles_1d_next)
        self.core_sources.advance()
        self.core_transport.advance()

        if do_refresh:
            return self.refresh()
        else:
            return core_profiles_1d_next

    def solve_15D(self, *args, tolerance=1.0e-4, max_iteration=1, **kwargs):
        self.equilibrium.refresh(
            core_profiles_1d=core_profiles_1d_iter,
            wall=self.wall,
            pf_active=self.pf_active,
            tf=self.tf,
            tolerance=tolerance,
            **kwargs,
        )

        residual = tolerance

        core_profiles_1d_iter = copy(self.core_profiles.profiles_1d.current)

        for step_num in range(max_iteration):
            equilibrium_time_slice = self.equilibrium.time_slice.current

            self.core_transport.refresh(
                equilibrium=equilibrium_time_slice,
                core_profile_1d=core_profiles_1d_iter,
            )

            core_transport_profiles_1d = (
                self.core_transport.model.combined.profiles_1d.current
            )

            self.core_sources.refresh(
                equilibrium=equilibrium_time_slice,
                core_profile_1d=core_profiles_1d_iter,
            )

            core_source_profiles_1d = (
                self.core_sources.source.combined.profiles_1d.current
            )

            core_profiles_1d_next = self.transport_solver.solve(
                equilibrium=equilibrium_time_slice,
                core_profiles_prev=core_profiles_1d_iter,
                core_transport_profiles_1d=core_transport_profiles_1d,
                core_source_profiles_1d=core_source_profiles_1d,
            )

            residual = self.check_converge(core_profiles_1d_iter, core_profiles_1d_next)

            if residual <= tolerance:
                break
            else:
                core_profiles_1d_iter = core_profiles_1d_next
        else:
            logger.debug(
                f"time={self.time}  iterator step {step_num}/{max_iteration} residual={residual}"
            )

        if residual >= tolerance:
            logger.warning(
                f"The solution does not converge, and the number of iterations exceeds the maximum {max_iteration}"
            )

        return core_profiles_1d_iter

    def refresh(self, /,
                core_profiles: CoreProfiles.TimeSlice,
                core_transport: CoreTransport.Model.TimeSlice,
                core_sources: CoreSources.Source.TimeSlice,
                equilibrium_prev: Equilibrium.TimeSlice,
                equilibrium_next: Equilibrium.TimeSlice = None,
                dt: float = None,
                **kwargs) -> CoreProfiles.TimeSlice:
        """
            solve transport equation until residual < tolerance
            return core_profiles
        """
        raise NotImplementedError("TransportSolverNumerics.refresh")

        # return CoreProfiles({
        #     "profiles_1d": {
        #         "grid": core_profiles_prev.profiles_1d.grid,
        #         "electrons": {"label": "e"},
        #         "ion": [{"label": ion.label} for ion in core_profiles_prev.profiles_1d.ion]
        #     }
        # })

    def advance(self, *args, **kwargs) -> CoreProfiles.TimeSlice:
        raise NotImplementedError("TransportSolverNumerics.advance")
