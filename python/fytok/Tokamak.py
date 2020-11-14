import collections
import copy
import math

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.Entry import open_entry
from spdm.util.AttributeTree import AttributeTree, _last_, _next_
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger

from .CoreProfiles import CoreProfiles
from .CoreSources import CoreSources
from .CoreTransports import CoreTransports
from .Equilibrium import Equilibrium
from .PFActive import PFActive
from .Transport import Transport
from .Wall import Wall


class Tokamak(AttributeTree):

    def __init__(self,  config=None,  *args,    **kwargs):
        super().__init__()

        if config is None:
            config = AttributeTree()

        self.wall = Wall(config.wall)
        self.pf_active = PFActive(config.pf_active)

        if not self.vacuum_toroidal_field.r0:  # FIXME: r0 should load from configure file
            lim_r = self.wall.limiter.outline.r
            self.vacuum_toroidal_field.r0 = (min(lim_r)+max(lim_r))*0.5

        self.equilibrium = Equilibrium(config.equilibrium, tokamak=self)
        self.core_profiles = CoreProfiles(config.core_profiles, tokamak=self)
        self.core_transports = CoreTransports(config.core_transports, tokamak=self)
        self.core_sources = CoreSources(config.core_sources, tokamak=self)
        self.transport = Transport(config.transport, tokamak=self)

    @staticmethod
    def load_imas(entry, itime=0):
        if isinstance(entry, str):
            entry = open_entry(entry)

        config = AttributeTree()
        config.wall = entry.wall
        config.pf_active = entry.pf_active
        config.equilibrium = entry.equilibrium.time_slice[itime].profiles_1d
        config.core_profiles = entry.core_profiles.profiles_1d[itime]

        for mode in entry.core_transports.mode:
            config.core_transports.mode[_next_].identifier = mode.identifier
            config.core_transports.mode[_last_].profiles_1d = mode.profiles_1d[itime]
        for source in entry.core_sources.mode:
            config.core_sources.load(source)
        return Tokamak(config)

    def save(self, uri, *args, **kwargs):
        raise NotImplementedError()

    def update(self, *args, time=0.0,
               core_profiles=None,
               ctx=None,
               fvec=1.0,
               constraints=None,
               max_iters=1,
               relative_deviation=0.1,
               ** kwargs):

        if core_profiles is not None:
            self.transport.core_profiles.update(core_profiles)

        convergence = False

        pressure_iter = self.transport.core_profiles.pressure

        for iter_count in range(max_iters):
            logger.debug(f"Iterator = {iter_count}")

            self.equilibrium.update(self.transport.core_profiles,
                                    time=time,
                                    fvec=fvec,
                                    constraints=constraints)

            self.core_transports.update(self.equilibrium, ctx=ctx)

            self.core_sources.update(self.equilibrium, ctx=ctx)

            self.transport.update(equilibrium=self.equilibrium,
                                  ctx=ctx,
                                  transports=self.core_transports,
                                  sources=self.core_sources)

            # TODO: edge
            # edge_profiles_old = copy(edge_profiles_iter)

            # edge_profiles_iter = self._transport_edge_solver(
            #     edge_profiles_old, dt,
            #     core_profiles_iter,
            #     equilibrium=self._equilibrium,
            #     transports=self.edge_transports,
            #     sources=self.edge_sources,
            #     **kwargs)

            if not pressure_iter:
                convergence = True
            # elif math.sqrt(sum(self.transport.core_profiles.pressure-pressure_iter**2) /
            #                sum(self.transport.core_profiles.pressure-pressure_iter**2)) < relative_deviation:
            #     convergence = True
            #     break

            pressure_iter = self.transport.core_profiles.pressure

        if not convergence:
            raise RuntimeError(f"Not convergence! iter_count={iter_count}")

    def plot(self, axis=None, *args,   **kwargs):

        if axis is None:
            axis = plt.gca()

        self.wall.plot(axis, **kwargs.get("wall", {}))
        self.pf_active.plot(axis, **kwargs.get("pf_active", {}))
        self.equilibrium.plot(axis, **kwargs.get("equilibrium", {}))

        axis.set_aspect('equal')
        # axis.axis('scaled')
        axis.set_xlabel(r"Major radius $R$ [m]")
        axis.set_ylabel(r"Height $Z$ [m]")
        axis.legend()

        return axis

    # def core_transports(self, *args,  **kwargs):
    #     """Core plasma transport of particles, energy, momentum and poloidal flux."""
    #     return NotImplemented

    # def core_sources(self, *args,  **kwargs):
    #     """Core plasma thermal source terms (for the transport equations of the thermal species).
    #     Energy terms correspond to the full kinetic energy equation
    #     (i.e. the energy flux takes into account the energy transported by the particle flux)
    #     """
    #     return NotImplemented

    # def edge_transports(self, *args,  **kwargs):
    #     """Edge plasma transport. Energy terms correspond to the full kinetic energy equation
    #      (i.e. the energy flux takes into account the energy transported by the particle flux)
    #     """
    #     return NotImplemented

    # def edge_sources(self, *args,  **kwargs):
    #     """Edge plasma sources. Energy terms correspond to the full kinetic energy equation
    #      (i.e. the energy flux takes into account the energy transported by the particle flux)"""
    #     return NotImplemented
