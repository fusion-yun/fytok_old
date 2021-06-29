
import collections
import datetime
import getpass
from typing import ChainMap, Union
from matplotlib.colors import to_rgb

import matplotlib.pyplot as plt
from spdm.data.Function import Function
from spdm.data.Node import Dict, Node, sp_property
from spdm.flow.Actor import Actor
from spdm.numlib import constants, np
from spdm.util.logger import logger

##################################
from .common.Misc import VacuumToroidalField
from .device.Magnetics import Magnetics
from .device.PFActive import PFActive
from .device.TF import TF
from .device.Wall import Wall
# ---------------------------------
from .transport.CoreProfiles import CoreProfiles
from .transport.CoreSources import CoreSources
from .transport.CoreTransport import CoreTransport
# ---------------------------------
from .transport.EdgeProfiles import EdgeProfiles
from .transport.EdgeSources import EdgeSources
from .transport.EdgeTransport import EdgeTransport
# ---------------------------------
from .transport.Equilibrium import Equilibrium
from .transport.MagneticCoordSystem import RadialGrid
from .transport.TransportSolver import TransportSolver

##################################
TWOPI = constants.pi*2.0


class Tokamak(Actor):
    """Tokamak
        功能：
            - 描述装置在单一时刻的状态，
            - 在时间推进时，确定各个子系统之间的依赖和演化关系，

    """

    def __init__(self, *args,  **kwargs):
        super().__init__(*args,  **kwargs)
        self._time = 0.0

    @property
    def time(self):
        return self._time

    @sp_property
    def radial_grid(self) -> RadialGrid:
        rho_tor_norm = self.get("radial_grid.rho_tor_norm", None)
        if rho_tor_norm is None:
            rho_tor_norm = np.linspace(0, 1.0, 128)
        return self.equilibrium.time_slice.radial_grid.remesh(rho_tor_norm, "rho_tor_norm")
    # --------------------------------------------------------------------------

    @sp_property
    def wall(self) -> Wall:
        return self.fetch("wall")

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
        return self.get("equilibrium")

    @sp_property
    def core_profiles(self) -> CoreProfiles:
        return CoreProfiles(self.get("core_profiles"), grid=self.radial_grid, parent=self)

    @sp_property
    def core_transport(self) -> CoreTransport:
        """Core plasma transport of particles, energy, momentum and poloidal flux."""
        return CoreTransport(self.get("core_transport"), grid=self.radial_grid, parent=self)

    @sp_property
    def core_sources(self) -> CoreSources:
        """Core plasma thermal source terms (for the transport equations of the thermal species).
            Energy terms correspond to the full kinetic energy equation
            (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return CoreSources(self.get("core_sources"), grid=self.radial_grid, parent=self)

    @sp_property
    def edge_profiles(self) -> EdgeProfiles:
        return self.get("edge_profiles")

    @sp_property
    def edge_transport(self) -> EdgeTransport:
        """Edge plasma transport. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return self.get("edge_transport")

    @sp_property
    def edge_sources(self) -> EdgeSources:
        """Edge plasma sources. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return self.get("edge_sources")

    @sp_property
    def transport_solver(self) -> TransportSolver:
        return self.get("transport_solver")

    def refresh(self, *args, **kwargs):
        super().refresh(*args, **kwargs)

    def advance(self,  dt=None, time=None, **kwargs):

        time = super().advance(time=time, dt=dt)

        self.wall.advance(time=time, refresh=False)

        self.pf_active.advance(time=time, refresh=False)

        self.equilibrium.advance(time=time, refresh=False)

        self.core_profiles.advance(time=time, refresh=False)

        self.core_sources.advance(time=time, refresh=False)

        self.core_transport.advance(time=time, refresh=False)

    def solve(self, *args, constraints: Equilibrium.Constraints = None, max_iteration=1, max_nodes=1250,  enable_edge=False,  tolerance=1.0e-6, **kwargs):

        for nstep in range(max_iteration):

            self.equilibrium.refresh(
                constraints=constraints,
                core_profiles=self.core_profiles,
                wall=self.wall,
                pf_active=self.pf_active,
                magnetics=self.magnetics)

            self.core_sources.refresh(equilibrium=self.equilibrium, core_profiles=self.core_profiles)

            self.core_transport.refresh(equilibrium=self.equilibrium, core_profiles=self.core_profiles)

            if enable_edge:
                self.edge_transport.refresh(equilibrium=self.equilibrium, core_profiles=self.core_profiles)

                self.edge_sources.refresh(equilibrium=self.equilibrium, core_profiles=self.core_profiles)

                self.edge_profiles.refresh()

            # Update grid
            self.core_profiles.refresh(equilibrium=self.equilibrium)

            # TODO: refresh boundary condition

            redisual = self.transport_solver.solve(equilibrium=self.equilibrium,
                                                   core_profiles=self.core_profiles,
                                                   core_sources=self.core_sources,
                                                   core_transport=self.core_transport,
                                                   max_nodes=max_nodes, tolerance=tolerance, **kwargs)

            logger.debug(f"time={self.time}  iterator step {nstep}/{max_iteration} redisual={redisual}")

            if redisual < tolerance:
                break

        if redisual > tolerance:
            logger.warning(
                f"The solution does not converge, and the number of iterations exceeds the maximum {max_iteration}")
        return redisual

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
