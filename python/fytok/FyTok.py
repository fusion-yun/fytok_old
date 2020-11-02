import collections
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from spdm.util.logger import logger

from .CoreTransport import CoreTransport
from .EdgeTransport import EdgeTransport
from .Equilibrium import Equilibrium
from .PFCoils import PFCoils
from .Wall import Wall


class FyTok:
    Profiles = collections.namedtuple("Profiles", "core  edge")

    def __init__(self, *args, equilibrium=None, tf_coils=None,  pf_coils=None, wall=None, **kwargs):
        super().__init__()
        # self._tf_coils = tf_coils or {}
        self._pf_coils = PFCoils(pf_coils)
        self._wall = Wall(wall)

        self._equilibrium = Equilibrium(self._wall, self._pf_coils)

        self._profiles = FyTok.Profiles({}, {})

        self._core_transport_solver = CoreTransport()
        self._edge_transport_solver = EdgeTransport()

        self._core_transports = None
        self._edge_transports = None

        self._core_sources = None
        self._edge_sources = None

    # @property
    # def tf_coils(self):
    #     return self._tf_coils

    @property
    def pf_coils(self):
        return self._pf_coils

    @property
    def wall(self):
        return self._wall

    @property
    def equilibrium(self):
        return self._equilibrium

    @property
    def profiles(self):
        if self._profiles is None:
            self._profiles = FyTok.Profiles(None, None)
        return self._profiles

    @property
    def core_profiles(self):
        return self.profiles.core

    @property
    def edge_profiles(self):
        return self.profiles.edge

    def core_transports(self, *args,  **kwargs):
        """Core plasma transport of particles, energy, momentum and poloidal flux."""
        return NotImplemented

    def core_sources(self, *args,  **kwargs):
        """Core plasma thermal source terms (for the transport equations of the thermal species).
        Energy terms correspond to the full kinetic energy equation
        (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return NotImplemented

    def edge_transports(self, *args,  **kwargs):
        """Edge plasma transport. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return NotImplemented

    def edge_sources(self, *args,  **kwargs):
        """Edge plasma sources. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)"""
        return NotImplemented

    def solve(self, dt,  *,   max_iters=100, ** kwargs):

        profiles_iter = copy(self.profiles)

        for iter_count in range(max_iters):

            self.equilibrium.solve(profiles=profiles_iter.core, **kwargs)

            profiles_old = copy(profiles_iter)

            profiles_iter = FyTok.Profiles(None, None)

            profiles_iter.core = self._core_transport_solver(
                profiles_old,
                dt,
                equilibrium=self._equilibrium,
                transports=self.core_transports,
                sources=self.core_sources,
                **kwargs)

            profiles_iter.edge = self._edge_transport_solver(
                FyTok.Profiles(profiles_iter.core, profiles_old.edge), dt,
                equilibrium=self._equilibrium,
                transports=self.edge_transports,
                sources=self.edge_sources,
                **kwargs)

            if self.check_convergence(profiles_old, profiles_iter):
                self._profiles = profiles_iter
                break
            elif iter_count == max_iters-1:
                raise RuntimeError(f"Too much iteration loop! count={iter_count}")

    def check_convergence(self, p_old, p_new):
        return False

    def plot(self, axis=None, **kwargs):

        if axis is None:
            axis = plt.gca()

        self.wall.plot(axis, **kwargs)
        self.pf_coils.plot(axis, **kwargs)
        self.equilibrium.plot(axis, **kwargs)

        axis.axis('scaled')
        return axis
