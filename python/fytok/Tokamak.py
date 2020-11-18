import collections
import copy
import math
from functools import cached_property
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
        self.load(config)
        self._time = 0
        # if config is None:
        #     config = AttributeTree()

        # self.vacuum_toroidal_field = config.vacuum_toroidal_field

        # if not self.vacuum_toroidal_field.r0:
        #     lim_r = self.wall.limiter.outline.r
        #     self.vacuum_toroidal_field.r0 = (min(lim_r)+max(lim_r))*0.5

    @staticmethod
    def load_from(entry, *args, **kwargs):
        return Tokamak(open_entry(entry, *args, **kwargs))

    def load(self, config):
        self._cache = config

    @property
    def time(self):
        return self._time

    @cached_property
    def vacuum_toroidal_field(self):
        r0 = float(self._cache.equilibrium.vacuum_toroidal_field.r0)
        b0 = float(self._cache.equilibrium.vacuum_toroidal_field.r0)

        if not r0:
            lim_r = self.wall.limiter.outline.r
            r0 = (min(lim_r)+max(lim_r))*0.5
        return AttributeTree(r0=r0, b0=b0)

    @cached_property
    def wall(self):
        return Wall(self._cache.wall)

    @cached_property
    def pf_active(self):
        return PFActive(self._cache.pf_active)

    @cached_property
    def equilibrium(self):
        return Equilibrium(self._cache.equilibrium.time_slice, tokamak=self)

    @cached_property
    def core_profiles(self):
        return CoreProfiles(self._cache.core_profiles, tokamak=self)

    @cached_property
    def core_transports(self):
        return CoreTransports(self._cache.core_transports.mode, tokamak=self)

    @cached_property
    def core_sources(self):
        return CoreSources(self._cache.core_sources.mode, tokamak=self)

    @cached_property
    def transport(self):
        return Transport(self._cache.transport, tokamak=self)

    def save(self, uri, *args, **kwargs):
        raise NotImplementedError()

    def update(self, *args, time=0.0,
               core_profiles=None,
               ctx=None,
               fvec=1.0,
               constraints=None,
               max_iters=1,
               tolerance=0.1,
               ** kwargs):

        if core_profiles is not None:
            self.transport.core_profiles.update(core_profiles)

        convergence = False

        pressure_iter = self.transport.core_profiles.pressure

        for iter_count in range(max_iters):
            logger.debug(f"Iterator = {iter_count}")

            self.equilibrium.update(profiles=self.transport.core_profiles, constraints=constraints)

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
        self.equilibrium.plot_profiles2d(axis, **kwargs.get("equilibrium", {}))

        axis.set_aspect('equal')
        axis.axis('scaled')
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
