

import matplotlib.pyplot as plt
from spdm.common.logger import logger
from spdm.data.Function import Function
from spdm.data.Node import Dict, List, Node, sp_property

from .common.Module import Module
# ---------------------------------
from .device.Magnetics import Magnetics
from .device.PFActive import PFActive
from .device.TF import TF
from .device.Wall import Wall
# ---------------------------------
from .transport.CoreProfiles import CoreProfiles
from .transport.CoreSources import CoreSources
from .transport.CoreTransport import CoreTransport
from .transport.CoreTransportSolver import CoreTransportSolver
# ---------------------------------
from .transport.EdgeProfiles import EdgeProfiles
from .transport.EdgeSources import EdgeSources
from .transport.EdgeTransport import EdgeTransport
from .transport.EdgeTransportSolver import EdgeTransportSolver
# ---------------------------------
from .transport.Equilibrium import Equilibrium
from .transport.EquilibriumSolver import EquilibriumSolver
from .transport.MagneticCoordSystem import RadialGrid


class Tokamak(Module):
    """Tokamak
        功能：
            - 描述装置在单一时刻的状态，
            - 在时间推进时，确定各个子系统之间的依赖和演化关系，

    """

    def __init__(self, *args,  **kwargs):
        super().__init__(*args,  **kwargs)

    @sp_property
    def wall(self) -> Wall:
        return self.get("wall")

    @sp_property
    def tf(self) -> TF:
        return self.get("tf")

    @sp_property
    def pf_active(self) -> PFActive:
        return self.get("pf_active")

    @sp_property
    def magnetics(self) -> Magnetics:
        return self.get("magnetics")
    # --------------------------------------------------------------------------

    @sp_property
    def equilibrium(self) -> Equilibrium:
        return self.get("equilibrium", {})

    @sp_property
    def core_profiles(self) -> CoreProfiles:
        return self.get("core_profiles")

    @sp_property
    def core_transport(self) -> CoreTransport:
        """Core plasma transport of particles, energy, momentum and poloidal flux."""
        return self.get("core_transport", {})

    @sp_property
    def core_sources(self) -> CoreSources:
        """
            Core plasma thermal source terms (for the transport equations of the thermal species).
            Energy terms correspond to the full kinetic energy equation
            (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return self.get("core_sources")

    @sp_property
    def edge_profiles(self) -> EdgeProfiles:
        return self.get("edge_profiles", {})

    @sp_property
    def edge_transport(self) -> EdgeTransport:
        """
            Edge plasma transport. Energy terms correspond to the full kinetic energy equation
            (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return self.get("edge_transport", {})

    @sp_property
    def edge_sources(self) -> EdgeSources:
        """Edge plasma sources. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return self.get("edge_sources", {})

    @sp_property
    def core_transport_solver(self) -> CoreTransportSolver:
        return self.get("core_transport_solver", {})

    @sp_property
    def edge_transport_solver(self) -> EdgeTransportSolver:
        return self.get("edge_transport_solver", {})

    @sp_property
    def equilibrium_solver(self) -> EquilibriumSolver:
        return self.get("equilibrium_solver", {})

    def check_converge(self, /, **kwargs):
        return 0.0

    def refresh(self, *args, time=None, tolerance=1.0e-4,   max_iteration=1, **kwargs) -> float:

        time_prev = self.time
        time_next = time if time is not None else time_prev

        dt = time_next-time_prev

        super().refresh(time=time_next)

        self.wall.refresh(time=time_next)

        self.pf_active.refresh(time=time_next)

        self.magnetics.refresh(time=time_next)

        core_profiles_prev = self.core_profiles
        core_profiles_iter = core_profiles_prev
        edge_profiles_prev = self.edge_profiles
        edge_profiles_iter = edge_profiles_prev

        equilibrium_prev = self.equilibrium

        var_list = []

        for step_num in range(max_iteration):

            equilibrium_iter = self.equilibrium_solver.solve(
                equilibrium_prev=equilibrium_prev,
                time=time,
                core_profiles=core_profiles_iter,
                edge_profiles=edge_profiles_iter,
                wall=self.wall,
                pf_active=self.pf_active,
                magnetics=self.magnetics)

            core_profiles_iter = self.core_transport_solver.solve(
                equilibrium_prev=equilibrium_prev,
                equilibrium_next=equilibrium_iter,
                core_profiles_prev=core_profiles_prev,
                core_sources=self.core_sources,
                core_transport=self.core_transport,
                dt=dt,
                var_list=var_list
            )

            edge_profiles_iter = self.edge_transport_solver.solve(
                equilibrium_prev=equilibrium_prev,
                equilibrium_next=equilibrium_iter,
                edge_profiles_prev=edge_profiles_prev,
                edge_sources=self.edge_sources,
                edge_transport=self.edge_transport,
                dt=dt
            )

            residual = self.check_converge(
                equilibrium_iter=equilibrium_iter,
                equilibrium_next=equilibrium_iter,
                core_profiles_iter=core_profiles_iter,
                core_profiles_prev=core_profiles_prev,
                edge_profiles_iter=edge_profiles_iter,
                edge_profiles_prev=edge_profiles_prev,
            )

            logger.debug(f"time={self.time}  iterator step {step_num}/{max_iteration} residual={residual}")

            if residual < tolerance:

                equilibrium_next = equilibrium_iter
                core_profiles_next = core_profiles_iter
                edge_profiles_next = edge_profiles_iter
                break

        self["equilibrium"] = equilibrium_next

        self["core_profiles"] = core_profiles_next

        self["edge_profiles"] = edge_profiles_next

        if residual > 1.0e-4:
            logger.warning(
                f"The solution does not converge, and the number of iterations exceeds the maximum {max_iteration}")

        return self.time

    def plot(self, axis=None, /,  **kwargs):

        if axis is None:
            axis = plt.gca()

        if kwargs.get("wall", True) is not False:
            self.wall.plot(axis, **kwargs.get("wall", {}))

        if kwargs.get("pf_active", True) is not False:
            self.pf_active.plot(axis, **kwargs.get("pf_active", {}))

        if kwargs.get("equilibrium", True) is not False:
            self.equilibrium.plot(axis,  **kwargs.get("equilibrium", {}))

        axis.set_aspect('equal')
        axis.axis('scaled')
        axis.set_xlabel(r"Major radius $R$ [m]")
        axis.set_ylabel(r"Height $Z$ [m]")
        # axis.legend()

        return axis

    # def initialize(self):
    #     r"""
    #         Set initial conditions self-consistently

    #     """

    #     gamma = self.equilibrium.profiles_1d.dvolume_drho_tor  \
    #         * self.equilibrium.profiles_1d.gm2    \
    #         / self.equilibrium.profiles_1d.fpol \
    #         * self.equilibrium.profiles_1d.dpsi_drho_tor \
    #         / (TWOPI**2)

    #     j_total = -gamma.derivative  \
    #         / self.equilibrium.profiles_1d.rho_tor[-1]**2 \
    #         * self.equilibrium.profiles_1d.dpsi_drho_tor  \
    #         * (self.equilibrium.profiles_1d.fpol**2) \
    #         / (constants.mu_0*self.vacuum_toroidal_field.b0) \
    #         * (constants.pi)

    #     j_total[1:] /= self.equilibrium.profiles_1d.dvolume_drho_tor[1:]
    #     j_total[0] = 2*j_total[1]-j_total[2]

    #     self.core_sources["j_parallel"] = j_total
