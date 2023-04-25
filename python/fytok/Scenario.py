from functools import cached_property

from spdm.data.Dict import Dict
from spdm.data.Node import Node
from spdm.utils.logger import logger

from .modules.PulseSchedule import PulseSchedule
from .modules.TransportSolverNumerics import TransportSolverNumerics
from .Tokamak import Tokamak


class Scenario(Dict[Node]):
    """
        Scenario

    """

    def __init__(self,  *args,  **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def tokamak(self):
        return Tokamak(self["tokamak"], parent=self)

    @cached_property
    def pulse_schedule(self):
        return PulseSchedule(self["pulse_schedule"], parent=self)

    @cached_property
    def transport_solver(self):
        return TransportSolverNumerics(self["transport_solver"], parent=self)

    # --------------------------------------------------------------------------

    def update(self, *args, time=None,   max_iters=1,  tolerance=0.1,   ** kwargs):

        convergence = False

        if time is None:
            time = self._time

        core_profiles_prev = self.core_profiles

        for iter_count in range(max_iters):
            logger.debug(f"Iterator = {iter_count}")

            for src in self.core_sources:
                src.update(time=time, equilibrium=self.equilibrium)

            for trans in self.core_transport:
                trans.update(time=time, equilibrium=self.equilibrium)

            core_profiles_next = self.transport.update(core_profiles_prev,
                                                       equilibrium=self.equilibrium,
                                                       core_transport=self.core_transport,
                                                       core_sources=self.core_sources,
                                                       boundary_condition=self.boundary_condition
                                                       )

            # .. todo:: integrate core and edge
            # edge_profiles_old = copy(edge_profiles_iter)

            # edge_profiles_iter = self._transport_edge_solver(
            #     edge_profiles_old, dt,
            #     core_profiles_next,
            #     equilibrium=self._equilibrium,
            #     transports=self.edge_transports,
            #     sources=self.edge_sources,
            #     **kwargs)

            if self.check_converge(core_profiles_prev, core_profiles_next, tolerance):
                convergence = True
                break

            core_profiles_prev = core_profiles_next

            self.equilibrium.update(time=time, profiles=core_profiles_next, constraints=self.constraints)

        if not convergence:
            raise RuntimeError(f"Does not converge! iter_count={iter_count}")
        else:
            self._core_profiles = core_profiles_next

    def check_converge(self, core_profiles_prev, core_profiles_next, tolerance):
        return True
