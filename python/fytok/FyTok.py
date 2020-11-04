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


class FyTok(AttributeTree):

    def __init__(self,   *args, **kwargs):
        super().__init__()
        if len(args)+len(kwargs) > 0:
            self.load(*args, **kwargs)

    def load(self, entry=None,  *args, itime=0,  backends={}, **kwags):
        if isinstance(entry, str):
            entry = open_entry(entry)

        self.entry.vacuum_toroidal_field
        self.entry.wall = Wall()
        self.entry.pf_active = PFActive()
        self.entry.core_profiles = CoreProfiles()
        self.entry.core_transports = CoreTransports(backend=backends.get("transport", None))
        self.entry.core_sources = CoreSources()

        self.entry.equilibrium = Equilibrium(backend=backends.get("equilibrium", None))

        self.transport = Transport(backend=backends.get("transport", None))

        # self._transport_edge_solver = TransportEdge(backend=backends.get("transport_edge", None))
        # self.edge_profiles = EdgeProfiles()
        # self.edge_tranports = EdgeTransport()
        # self.edge_sources = EdgeSources()

        if isinstance(entry, LazyProxy):
            self.entry.wall.load(entry.wall)
            self.entry.pf_active.load(entry.pf_active)
            self.entry.equilibrium.load(entry.equilibrium.time_slice[itime],  tokamak=self.entry)
            # self.entry.core_profiles.load(entry.core_profiles.profiles_1d[itime])
            # self.entry.core_transports.load(entry.core_transports.model.profiles_1d[itime])
            # self.entry.core_sources.load(entry.core_sources.source[itime])

        else:
            self.entry.wall.load(**kwags.get("wall", {}))
            self.entry.pf_active.load(**kwags.get("pf_active", {}))
            self.entry.equilibrium.load(**kwags.get("equilibrium", {}), tokamak=self.entry)
            # self.entry.core_profiles.load(**kwags.get("core_profiles", {}))
            # self.entry.core_transports.load(**kwags.get("core_transports", {}))
            # self.entry.core_sources.load(**kwags.get("core_sources", {}))


        # self.entry.wall.limiter = [self._data_source.wall.description_2d[0].limiter.unit[0].outline.r.__value__(),
        #                            self._data_source.wall.description_2d[0].limiter.unit[0].outline.z.__value__()]
        # self.entry.wall.vessel.inner = [self._data_source .wall.description_2d.vessel.annular.outline_inner.r.__value__(),
        #                                 self._data_source .wall.description_2d.vessel.annular.outline_inner.z.__value__()]
        # self.entry.wall.vessel.outer = [self._data_source .wall.description_2d.vessel.annular.outline_outer.r.__value__(),
        #                                 self._data_source .wall.description_2d.vessel.annular.outline_outer.z.__value__()]
        # for coil in self._data_source.pf_active.coil:
        #     rect = coil.element[0].geometry.rectangle.__value__()
        #     self.entry.pf_active.coil.__push_back__(
        #         {
        #             "name": str(coil.name),
        #             "r": float(rect.r),
        #             "z": float(rect.z),
        #             "width": float(rect.width),
        #             "height": float(rect.height),
        #             "turns": int(coil.element[0].turns_with_sign)
        #         }
        #     )
        # try:
        #     itime = kwargs.get("itime", 0)
        #     lfcs_r = self._data_source .equilibrium.time_slice[itime].boundary.outline.r.__value__()[:, 0]
        #     lfcs_z = self._data_source .equilibrium.time_slice[itime].boundary.outline.z.__value__()[:, 0]

        #     psivals = [(R, Z, 0.0) for R, Z in zip(lfcs_r, lfcs_z)]
        #     psivals = [(R, Z, 0.0) for R, Z in zip(self._data_source .equilibrium.time_slice[itime].boundary.outline.r.__value__(),
        #                                            self._data_source .equilibrium.time_slice[itime].boundary.outline.z.__value__())]

        # except KeyError:
        #     pass
        # else:
        #     self.equilibrium.solve(psivals=psivals)

    def save(self, uri, *args, **kwargs):
        raise NotImplementedError()


    def solve(self, dt,  *,   max_iters=100,   ** kwargs):

        core_profiles_iter = self.entry.core_profiles

        for iter_count in range(max_iters):

            self.equilibrium.solve(core_profiles_iter, **kwargs)

            core_profiles_new = self.transport.solve(
                core_profiles_iter, dt,
                equilibrium=self.equilibrium,
                transports=self.core_transports,
                sources=self.core_sources,
                **kwargs)

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
                self.entry.core_profiles = core_profiles_new
                # self._edge_profiles = edge_profiles_iter
                break
            elif iter_count == max_iters-1:
                raise RuntimeError(f"Too much iteration loop! count={iter_count}")
            else:
                core_profiles_iter = core_profiles_new

    def check_convergence(self, p_old, p_new):
        return False

    def plot(self, axis=None, **kwargs):

        if axis is None:
            axis = plt.gca()

        self.entry.wall.plot(axis, **kwargs)
        self.entry.pf_active.plot(axis, **kwargs)
        self.entry.equilibrium.plot(axis, **kwargs)

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
