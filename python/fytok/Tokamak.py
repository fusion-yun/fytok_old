
import collections
import datetime
import getpass
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.constants
from spdm.data.AttributeTree import as_attribute_tree
from spdm.data.Function import Function
from spdm.data.Node import Dict
from spdm.util.logger import logger

from .modules.device.PFActive import PFActive
from .modules.device.TF import TF
from .modules.device.Wall import Wall
from .modules.transport.CoreProfiles import CoreProfiles
from .modules.transport.CoreSources import CoreSources
from .modules.transport.CoreTransport import CoreTransport
from .modules.transport.EdgeProfiles import EdgeProfiles
from .modules.transport.EdgeSources import EdgeSources
from .modules.transport.EdgeTransport import EdgeTransport
from .modules.transport.Equilibrium import Equilibrium
from .modules.transport.MagneticCoordSystem import RadialGrid
from .modules.transport.TransportSolver import TransportSolver
from .modules.utilities.Misc import VacuumToroidalField

TWOPI = scipy.constants.pi*2.0


class Tokamak(Dict):
    """Tokamak
        功能：
            - 描述装置在单一时刻的状态，
            - 在时间推进时，确定各个子系统之间的依赖和演化关系，

    """

    def __init__(self, *args,  radial_grid=None, **kwargs):
        super().__init__(*args,  **kwargs)
        self._radial_grid = radial_grid

    @property
    def time(self):
        return self._time

    @cached_property
    def wall(self) -> Wall:
        return Wall(self["wall"], parent=self)

    @cached_property
    def tf(self) -> TF:
        return TF(self["tf"], parent=self)

    @cached_property
    def pf_active(self) -> PFActive:
        return PFActive(self["pf_active"], parent=self)

    # --------------------------------------------------------------------------

    @cached_property
    def boundary_conditions(self):
        return Dict(self["boundary_conditions"], parent=self)

    @cached_property
    def constraints(self):
        return Dict(self["constraints"], parent=self)

    @cached_property
    def equilibrium(self) -> Equilibrium:
        return Equilibrium(self["equilibrium"], parent=self)

        #    vacuum_toroidal_field=self.vacuum_toroidal_field,
        #    constraints=self.constraints,
        #    wall=self.wall,
        #    pf_active=self.pf_active,
        #    tf=self.tf,

    @cached_property
    def core_profiles(self) -> CoreProfiles:
        return CoreProfiles(self["core_profiles"],  parent=self)

    @cached_property
    def core_transport(self) -> CoreTransport:
        """Core plasma transport of particles, energy, momentum and poloidal flux."""
        return CoreTransport(self["core_transport"],  parent=self)

    @cached_property
    def core_sources(self) -> CoreSources:
        """Core plasma thermal source terms (for the transport equations of the thermal species).
            Energy terms correspond to the full kinetic energy equation
            (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return CoreSources(self["core_sources"],  parent=self)

    @cached_property
    def edge_profiles(self) -> EdgeProfiles:
        return EdgeProfiles(self["edge_profiles"], parent=self)

    @cached_property
    def edge_transport(self) -> EdgeTransport:
        """Edge plasma transport. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return EdgeTransport(self["edge_transport"], parent=self)

    @cached_property
    def edge_sources(self) -> EdgeSources:
        """Edge plasma sources. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return EdgeSources(self["edge_sources"], parent=self)

    @cached_property
    def transport_solver(self) -> TransportSolver:
        return TransportSolver(self["transport_solver"], parent=self)

    def update(self,
               time=None,
               vacuum_toroidal_field=None,
               constraints=None,
               equilibrium=None,
               core_transport=None,
               core_sources=None,
               boundary_conditions=None,
               tolerance=1.0e-6,
               max_step=1,
               **kwargs):

        if time is not None:
            time = self._time

        if vacuum_toroidal_field is not None:
            del self.vacuum_toroidal_field
            self["vacuum_toroidal_field"] = vacuum_toroidal_field

        if constraints is not None:
            del self.constraints
            self["constraints"] = constraints

        if equilibrium is not None:
            del self.equilibrium
            self["equilibrium"] = equilibrium

        if core_transport is not None:
            # del self.core_transport
            self["core_transport"] = core_transport

        if core_sources is not None:
            # del self.core_sources
            self["core_sources"] = core_sources

        if boundary_conditions is not None:
            # del self.boundary_conditions
            self["boundary_conditions"] = boundary_conditions

        core_profiles_prev = self.core_profiles
        core_profiles_next = CoreProfiles(grid=core_profiles_prev.profiles_1d.grid)
        for nstep in range(max_step):
            logger.debug(f"time={time}  iterator step {nstep}/{max_step}")

            # self.radial_grid.update(time=time, equilibrium=self.equilibrium)

            # self.core_transport.update(time=time,
            #                            equilibrium=self.equilibrium,
            #                            core_profiles=self.core_profiles)

            # self.core_sources.update(time=time,
            #                          equilibrium=self.equilibrium,
            #                          core_profiles=self.core_profiles)

            # TODO: using EdgeProfile update  self.boundary_conditions

            self.transport_solver.solve(
                core_profiles_prev,
                core_profiles_next,
                time=time,
                equilibrium=self.equilibrium,
                vacuum_toroidal_field=self.vacuum_toroidal_field,
                core_transport=self.core_transport,
                core_sources=self.core_sources,
                boundary_conditions=self.boundary_conditions,
                tolerance=tolerance,
                verbose=0,
                **kwargs.get("transport_solver", {})
            )

            # TODO: Update Edge

            logger.warning(f"TODO: EdgeTransport")

            # self.edge_transport.update(time=time, equilibrium=self.equilibrium, core_profiles_prev=self.core_profiles)

            # self.edge_sources.update(time=time, equilibrium=self.equilibrium, core_profiles_prev=self.core_profiles)

            self.constraints.update(time=time)

            if self.equilibrium.update(
                vacuum_toroidal_field=self.vacuum_toroidal_field,
                psi=core_profiles_next.profiles_1d.psi,
                constraints=self.constraints,
                core_profiles=core_profiles_next,
                test_convergence=True
            ):
                break

            core_profiles_prev = core_profiles_next

        self.__dict__["core_profiles"] = core_profiles_next
        self._time = time

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
    #         / (scipy.constants.mu_0*self.vacuum_toroidal_field.b0) \
    #         * (scipy.constants.pi)

    #     j_total[1:] /= self.equilibrium.profiles_1d.dvolume_drho_tor[1:]
    #     j_total[0] = 2*j_total[1]-j_total[2]

    #     self.core_sources["j_parallel"] = j_total
