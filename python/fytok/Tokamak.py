import collections
import copy

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.Entry import open_entry
from spdm.util.AttributeTree import AttributeTree
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger

from .CoreProfiles import CoreProfiles
from .CoreSources import CoreSources
from .CoreTransports import CoreTransports
from .Equilibrium import Equilibrium
from .PFActive import PFActive
from .Wall import Wall

from .Transport import Transport


class Tokamak(AttributeTree):

    def __init__(self,   *args, config=None, backends={}, **kwargs):
        super().__init__()
        self.wall = Wall()
        self.pf_active = PFActive()
        self.core_profiles = CoreProfiles()
        self.core_transports = CoreTransports()
        self.core_sources = CoreSources()
        self.equilibrium = Equilibrium(backend=backends.get("equilibrium", None), tokamak=self)
        self.transport = Transport(backend=backends.get("transport", None))

        if config is not None:
            self.load(config)

    def load(self, config, itime=0):
        if isinstance(config, str):
            config = open_entry(config)

        # self._transport_edge_solver = TransportEdge(backend=backends.get("transport_edge", None))
        # self.edge_profiles = EdgeProfiles()
        # self.edge_tranports = EdgeTransport()
        # self.edge_sources = EdgeSources()
        if isinstance(config, LazyProxy):

            # self.core_profiles.load(config.core_profiles)
            # self.core_transports.load(config.core_transports)
            # self.core_sources.load(config.core_sources)
            self.vacuum_toroidal_field.r0 = config.equilibrium.vacuum_toroidal_field.r0() or 1.0
            self.vacuum_toroidal_field.b0 = config.equilibrium.vacuum_toroidal_field.b0[itime]() or 1.0

            self.wall.load(config.wall)
            self.pf_active.load(config.pf_active)
            self.equilibrium.load(config.equilibrium)
        else:
            self.vacuum_toroidal_field.r0 = config.get("R0") or self.wall.limiter.outline.r.mean()
            self.vacuum_toroidal_field.b0 = config.get("B0") or 1.0

            self.wall.load(config.get("wall", {}))
            self.pf_active.load(config.get("pf_active", {}))
            self.equilibrium.load(config.get("equilibrium", {}), tokamak=self)
            self.core_profiles.load(config.get("core_profiles", {}))
            self.core_transports.load(config.get("core_transports", {}))
            self.core_sources.load(config.get("core_sources", {}))

            self.core_profiles.vacuum_toroidal_field.r0 = self.vacuum_toroidal_field.r0
            self.core_profiles.vacuum_toroidal_field.b0 = self.vacuum_toroidal_field.b0

        return self

    def save(self, uri, *args, **kwargs):
        raise NotImplementedError()

    def solve(self, dt, *,   constraints=None, **kwargs):

        fvec = self.vacuum_toroidal_field.r0 * self.vacuum_toroidal_field.b0

        core_profiles_iter = self.core_profiles

        for iter_count in range(max_iters):

            self.equilibrium.solve(core_profiles_iter, fvec=fvec,   constraints=constraints)

            core_profiles_new = self.transport.solve(
                core_profiles_iter, dt,
                equilibrium=self.equilibrium,
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

            if self.check_convergence(core_profiles_new, core_profiles_iter):
                self.core_profiles = core_profiles_new
                # self._edge_profiles = edge_profiles_iter
                break
            elif iter_count == max_iters-1:
                raise RuntimeError(f"Too much iteration loop! count={iter_count}")
            else:
                core_profiles_iter = core_profiles_new

    def check_convergence(self, p_old, p_new):
        return False

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
