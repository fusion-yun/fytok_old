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

    def __init__(self,   *args, **kwargs):
        super().__init__()
        if len(args)+len(kwargs) > 0:
            self.load(*args, **kwargs)

    def load(self, entry=None,  *args, itime=0, R0=None, B0=None, backends={}, **kwargs):
        if isinstance(entry, str):
            entry = open_entry(entry)

        self.vacuum_toroidal_field.r0 = kwargs.get("R0", 1.0)
        self.vacuum_toroidal_field.b0 = kwargs.get("B0", 1.0)

        self.wall = Wall()
        self.pf_active = PFActive()
        self.core_profiles = CoreProfiles()
        self.core_transports = CoreTransports(backend=backends.get("transport", None))
        self.core_sources = CoreSources()

        self.equilibrium = Equilibrium(backend=backends.get("equilibrium", None))

        self.transport = Transport(backend=backends.get("transport", None))

        # self._transport_edge_solver = TransportEdge(backend=backends.get("transport_edge", None))
        # self.edge_profiles = EdgeProfiles()
        # self.edge_tranports = EdgeTransport()
        # self.edge_sources = EdgeSources()

        if isinstance(entry, LazyProxy):
            self.wall.load(entry.wall)
            self.pf_active.load(entry.pf_active)
            self.equilibrium.load(entry.equilibrium.time_slice[itime](),  tokamak=self)
            self.core_profiles.load(entry.core_profiles.profiles_1d[itime])
            # self.core_transports.load(core_transports.model.profiles_1d[itime])
            # self.core_sources.load(core_sources.source[itime])
            self.vacuum_toroidal_field.r0 = entry.equilibrium.vacuum_toroidal_field.r0() or 1.0
            self.vacuum_toroidal_field.b0 = entry.equilibrium.vacuum_toroidal_field.b0[itime]() or 1.0
        else:
            self.wall.load(**kwargs.get("wall", {}))
            self.pf_active.load(**kwargs.get("pf_active", {}))
            self.equilibrium.load(**kwargs.get("equilibrium", {}), tokamak=self)
            self.core_profiles.load(**kwargs.get("core_profiles", {}))
            # self.core_transports.load(**kwargs.get("core_transports", {}))
            # self.core_sources.load(**kwargs.get("core_sources", {}))

        self.vacuum_toroidal_field.r0 = R0 or self.wall.limiter.outline.r.mean()
        self.vacuum_toroidal_field.b0 = B0 or 1.0
        self.core_profiles.vacuum_toroidal_field.r0 = self.vacuum_toroidal_field.r0
        self.core_profiles.vacuum_toroidal_field.b0 = self.vacuum_toroidal_field.b0
        return self

    def save(self, uri, *args, **kwargs):
        raise NotImplementedError()

    def solve(self, dt, *, max_iters=100,  B0=None,  **constraints):

        if B0 is not None:
            self.vacuum_toroidal_field.b0 = B0

        fvec = self.vacuum_toroidal_field.r0 * self.vacuum_toroidal_field.b0

        core_profiles_iter = self.core_profiles

        for iter_count in range(max_iters):

            # self.equilibrium.solve(core_profiles_iter, fvec=fvec,  **constraints)

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

    def plot_full(self,  profiles=None, profiles_label=None, x_axis="psi_norm", xlabel=r'$\psi_{norm}$', *args, **kwargs):

        if isinstance(profiles, str):
            profiles.split(" ")
        elif profiles is None:
            profiles = ["q", "pprime", "ffprime", "fpol", "pressure"]
            profiles_label = [r"q", r"$p^{\prime}$",  r"$f f^{\prime}$", r"$f_{pol}$", r"pressure"]

        nprofiles = len(profiles)
        if nprofiles == 0:
            return self.plot(*args, **kwargs)

        fig, axs = plt.subplots(ncols=2, nrows=nprofiles, sharex=True)
        gs = axs[0, 1].get_gridspec()
        # remove the underlying axes
        for ax in axs[:, 1]:
            ax.remove()
        ax_right = fig.add_subplot(gs[:, 1])

        self.plot(ax_right, *args, **kwargs)

        x = self.equilibrium.profiles_1d[x_axis]
        #  np.linspace(0, 1.0, 16)

        if profiles_label is None:
            profiles_label = profiles

        for idx, pname in enumerate(profiles):
            y = self.equilibrium.profiles_1d[pname](x)
            axs[idx, 0].plot(x, y,  label=profiles_label[idx])
            # axs[idx, 0].set_ylabel(profiles_label[idx])
            axs[idx, 0].legend()

        axs[nprofiles-1, 0].set_xlabel(xlabel)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        return fig

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
