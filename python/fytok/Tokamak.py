
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.PhysicalGraph import PhysicalGraph
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
from .util.RadialGrid import RadialGrid


class Tokamak(PhysicalGraph):
    """Tokamak
        功能：
            - 描述装置在单一时刻的状态，
            - 在时间推进时，确定各个子系统之间的依赖和演化关系，

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args,  **kwargs)

        # self._time = PhysicalGraph(self["equilibrium.time"].__fetch__())
        # self._vacuum_toroidal_field = PhysicalGraph(self["equilibrium.vacuum_toroidal_field"].__fetch__())

    # --------------------------------------------------------------------------

    @property
    def time(self):
        return self._time

    @property
    def vacuum_toroidal_field(self):
        return self._vacuum_toroidal_field

    @cached_property
    def wall(self):
        return Wall(self["wall.description_2d"], parent=self)

    @cached_property
    def tf(self):
        return TF(self["tf"], parent=self)

    @cached_property
    def pf_active(self):
        return PFActive(self["pf_active"], parent=self)

    # --------------------------------------------------------------------------

    @cached_property
    def equilibrium(self):
        eq = self["equilibrium"]
        if eq["time_slice"] != None:
            eq = eq["time_slice"]
        return Equilibrium(eq, parent=self)

    @cached_property
    def core_profiles(self):
        return CoreProfiles(self["core_profiles"],   parent=self)

    @cached_property
    def edge_profiles(self):
        return EdgeProfiles(self["edge_profiles"],   parent=self)

    @cached_property
    def core_transport(self):
        """Core plasma transport of particles, energy, momentum and poloidal flux."""
        return CoreTransport(self["core_transport"],   parent=self)

    @cached_property
    def core_sources(self):
        """Core plasma thermal source terms (for the transport equations of the thermal species).
            Energy terms correspond to the full kinetic energy equation
            (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return CoreSources(self["core_sources"],   parent=self)

    @cached_property
    def edge_transports(self):
        """Edge plasma transport. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return EdgeTransport(self["edge_transport.mode"], parent=self)

    @cached_property
    def edge_sources(self):
        """Edge plasma sources. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return CoreSources(self["edge_sources.mode"], parent=self)

    @cached_property
    def constraints(self):
        return PhysicalGraph(self["constraints"], parent=self)

    def plot(self, axis=None, *args,   **kwargs):

        if axis is None:
            axis = plt.gca()

        if kwargs.get("wall", True) is not False:
            self.wall.plot(axis, **kwargs.get("wall", {}))

        if kwargs.get("pf_active", True) is not False:
            self.pf_active.plot(axis, **kwargs.get("pf_active", {}))

        if kwargs.get("equilibrium", True) is not False:
            self.equilibrium.plot(axis, **kwargs.get("equilibrium", {}))

        axis.set_aspect('equal')
        axis.axis('scaled')
        axis.set_xlabel(r"Major radius $R$ [m]")
        axis.set_ylabel(r"Height $Z$ [m]")
        # axis.legend()
        return axis
