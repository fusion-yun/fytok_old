
import collections
import datetime
import getpass
from typing import Union
from spdm.util.numlib import np
import matplotlib.pyplot as plt
from fytok.modules.transport.MagneticCoordSystem import RadialGrid
from spdm.util.numlib import constants
from spdm.data.Function import Function
from spdm.data.Node import Dict, Node
from spdm.data.Node import sp_property
from spdm.flow.Actor import Actor
from spdm.util.logger import logger

##################################
from .modules.common.Misc import VacuumToroidalField
from .modules.device.Magnetics import Magnetics
from .modules.device.PFActive import PFActive
from .modules.device.TF import TF
from .modules.device.Wall import Wall
# ---------------------------------
from .modules.transport.CoreProfiles import CoreProfiles
from .modules.transport.CoreSources import CoreSources
from .modules.transport.CoreTransport import CoreTransport
# ---------------------------------
from .modules.transport.EdgeProfiles import EdgeProfiles
from .modules.transport.EdgeSources import EdgeSources
from .modules.transport.EdgeTransport import EdgeTransport
# ---------------------------------
from .modules.transport.Equilibrium import Equilibrium
from .modules.transport.TransportSolver import TransportSolver

##################################
TWOPI = constants.pi*2.0


class Tokamak(Actor):
    """Tokamak
        功能：
            - 描述装置在单一时刻的状态，
            - 在时间推进时，确定各个子系统之间的依赖和演化关系，

    """

    def __init__(self, d=None, * args, grid: Union[RadialGrid, np.ndarray, None] = None, **kwargs):
        super().__init__(kwargs)
        self._time = 0.0
        self._grid = grid

    @property
    def time(self):
        return self._time

    @property
    def grid(self):
        if not isinstance(self._grid, RadialGrid):
            self._grid = self.equilibrium.time_slice.coordinate_system.radial_grid(
                self._grid, primary_axis="rho_tor_norm")
        return self._grid
    # --------------------------------------------------------------------------

    @sp_property
    def wall(self) -> Wall:
        return self["wall"]

    @sp_property
    def tf(self) -> TF:
        return self["tf"]

    @sp_property
    def pf_active(self) -> PFActive:
        return self["pf_active"]

    @sp_property
    def magnetics(self) -> Magnetics:
        return self["magnetics"]
    # --------------------------------------------------------------------------

    @sp_property
    def equilibrium(self) -> Equilibrium:
        return self["equilibrium"]

    @sp_property
    def core_profiles(self) -> CoreProfiles:
        return self["core_profiles"]

    @sp_property
    def core_transport(self) -> CoreTransport:
        """Core plasma transport of particles, energy, momentum and poloidal flux."""
        return self["core_transport"]

    @sp_property
    def core_sources(self) -> CoreSources:
        """Core plasma thermal source terms (for the transport equations of the thermal species).
            Energy terms correspond to the full kinetic energy equation
            (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return self["core_sources"]

    @sp_property
    def edge_profiles(self) -> EdgeProfiles:
        return self["edge_profiles"]

    @sp_property
    def edge_transport(self) -> EdgeTransport:
        """Edge plasma transport. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return self["edge_transport"]

    @sp_property
    def edge_sources(self) -> EdgeSources:
        """Edge plasma sources. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return self["edge_sources"]

    @sp_property
    def transport_solver(self) -> TransportSolver:
        return self["transport_solver"]

    def advance(self, time=None,  dt=None,  vacuum_toroidal_field: VacuumToroidalField = None, **kwargs):

        time = super().advance(time=time, dt=dt)

        vacuum_toroidal_field = VacuumToroidalField(r0=self._r0, b0=self._b0)

        self.pf_active.advance(time=time)

        self.equilibrium.advance(time=time, vacuum_toroidal_field=vacuum_toroidal_field,
                                 wall=self.wall, pf_active=self.pf_active.current_state,)

        self.core_profiles.advance(time=time)

        self.core_sources.advance(time=time)

        self.core_transport.advance(time=time)

        self.update(time=time, vacuum_toroidal_field=vacuum_toroidal_field, **kwargs)

    def update(self,  vacuum_toroidal_field: VacuumToroidalField = None, max_step=1, enable_edge=False, tolerance=1.0e-6, **kwargs):

        if vacuum_toroidal_field is not None:
            self._b0[-1] = vacuum_toroidal_field.b0

        vacuum_toroidal_field = VacuumToroidalField(r0=self._r0, b0=self._b0[-1])

        core_profile_prev = self.core_profiles.previous_state
        core_profile_next = self.core_profiles.current_state

        for nstep in range(max_step):
            equilibrium = self.equilibrium.current_state

            self.core_transport.update(vacuum_toroidal_field=vacuum_toroidal_field,
                                       equilibrium=equilibrium, core_profiles=core_profile_next)

            self.core_sources.update(vacuum_toroidal_field=vacuum_toroidal_field,
                                     equilibrium=equilibrium, core_profiles=core_profile_next)

            convergence = self.transport_solver.update(
                core_profile_prev,  core_profile_next,
                vacuum_toroidal_field=vacuum_toroidal_field,
                equilibrium=equilibrium,
                core_transport=self.core_transport.current_state,
                core_sources=self.core_sources.current_state,
                **kwargs)

            if enable_edge:
                self.edge_transport.update(vacuum_toroidal_field=vacuum_toroidal_field,
                                           equilibrium=equilibrium, core_profiles=core_profile_next)

                self.edge_sources.update(vacuum_toroidal_field=vacuum_toroidal_field,
                                         equilibrium=equilibrium, core_profiles=core_profile_next)

                self.edge_profiles.update(vacuum_toroidal_field=vacuum_toroidal_field,
                                          equilibrium=equilibrium, core_profiles=core_profile_next,
                                          edge_transport=self.edge_transport.current_state,
                                          edge_sources=self.edge_sources.current_state)

                # TODO: update boundary condition

            self.equilibrium.update(vacuum_toroidal_field=vacuum_toroidal_field,
                                    wall=self.wall,
                                    pf_active=self.pf_active.current_state,
                                    core_profiles=core_profile_next)

            logger.debug(f"time={self.time}  iterator step {nstep}/{max_step} convergence={convergence}")

            if convergence < tolerance:
                break

    def plot(self, axis=None, *args, title=None, time=None,  **kwargs):

        if axis is None:
            axis = plt.gca()

        if kwargs.get("wall", True) is not False:
            self.wall.plot(axis, **kwargs.get("wall", {}))

        if kwargs.get("pf_active", True) is not False:
            self.pf_active.plot(axis, **kwargs.get("pf_active", {}))

        if kwargs.get("equilibrium", True) is not False:
            self.equilibrium.plot(axis, time=time,  **kwargs.get("equilibrium", {}))

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
