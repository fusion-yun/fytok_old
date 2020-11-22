import collections
import copy
import math
from functools import cached_property
import matplotlib.pyplot as plt
import numpy as np
from spdm.util.AttributeTree import AttributeTree, _last_, _next_
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger

from fytok.CoreProfiles import CoreProfiles
from fytok.CoreSources import CoreSources
from fytok.CoreTransports import CoreTransports
from fytok.Equilibrium import Equilibrium
from fytok.PFActive import PFActive
from fytok.TransportSolver import TransportSolver
from fytok.Wall import Wall
from fytok.TF import TF


class Tokamak(AttributeTree):

    def __init__(self,  cache=None,  *args, time=0.0, core_profiles=None,  **kwargs):
        super().__init__(*args, time=time, **kwargs)
        if isinstance(cache, LazyProxy) or isinstance(cache, AttributeTree):
            self._cache = cache
        else:
            self._cache = AttributeTree(cache)
        self.core_profiles = CoreProfiles(
            core_profiles or self._cache.core_profiles,
            time=self.time, equilibrium=self.equilibrium)

    @cached_property
    def vacuum_toroidal_field(self):
        r0 = float(self._cache.equilibrium.vacuum_toroidal_field.r0)
        b0 = float(self._cache.equilibrium.vacuum_toroidal_field.b0)

        if not r0:
            lim_r = self.wall.limiter.outline.r
            r0 = (min(lim_r)+max(lim_r))*0.5
        return AttributeTree(r0=r0, b0=b0)

    @cached_property
    def wall(self):
        return Wall(self._cache.wall, tokamak=self)

    @cached_property
    def tf(self):
        return TF(self._cache.tf, tokamak=self)

    @cached_property
    def pf_active(self):
        return PFActive(self._cache.pf_active, tokamak=self)

    @cached_property
    def equilibrium(self):
        return Equilibrium(self._cache.equilibrium.time_slice, tokamak=self)

    @cached_property
    def core_transports(self):
        return CoreTransports(self._cache.core_transports.mode, tokamak=self)

    @cached_property
    def core_sources(self):
        return CoreSources(self._cache.core_sources.mode, tokamak=self)

    @cached_property
    def transport(self):
        return TransportSolver(self._cache.transport, tokamak=self)

    def save(self, uri, *args, **kwargs):
        raise NotImplementedError()

    def update(self, *args,
               time=0.0,
               core_profiles=None,
               constraints=None,
               max_iters=1,
               tolerance=0.1,
               ** kwargs):

        convergence = False

        core_profiles_iter = CoreProfiles(core_profiles or {}, equilibrium=self.equilibrium)
        core_profiles_prev = self.core_profiles

        for iter_count in range(max_iters):
            logger.debug(f"Iterator = {iter_count}")

            # self.equilibrium.update(profiles=self.core_profiles.interploate(["pprime", "ffprime"]),
            #                         constraints=constraints)

            core_profiles_iter = CoreProfiles(
                core_profiles,
                equilibrium=self.equilibrium,
                rho_tor_norm=core_profiles_prev.grid.rho_tor_norm)

            self.core_sources.update()

            self.core_transports.update()

            tol = self.transport.update(
                core_profiles_prev,
                core_profiles_iter,
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

            if tol < tolerance:
                convergence = True
                break
            else:
                core_profiles_prev = core_profiles_iter

        if not convergence:
            raise RuntimeError(f"Does not converge! iter_count={iter_count}")
        else:
            self.core_profiles = core_profiles_iter

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
