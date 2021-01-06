import collections
import copy
import math
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma
import scipy.integrate
from spdm.util.AttributeTree import AttributeTree, _last_, _next_
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.data.Profile import Profile

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
from .modules.transport.TransportSolver import TransportSolver
from .util.RadialGrid import RadialGrid


class Tokamak(AttributeTree):
    """Tokamak
        功能：
            - 描述装置在单一时刻的状态，
            - 在时间推进时，确定各个子系统之间的依赖和演化关系，

    """

    def __init__(self,  cache=None,  *args, time=0.0, rho_tor_norm=None,   **kwargs):
        super().__init__(*args, time=time, **kwargs)
        self.__dict__["_cache"] = cache or AttributeTree()
        self.__dict__["_time"] = time

        self._time = time
        if rho_tor_norm is None:
            rho_tor_norm = np.sqrt(np.linspace(0, 1.0, 129))
        else:
            rho_tor_norm = rho_tor_norm

        self._grid = RadialGrid(rho_tor_norm, equilibrium=self.equilibrium)

        self._core_profiles = None

        self._edge_profiles = None

    # --------------------------------------------------------------------------
    @property
    def time(self):
        return self._time

    @property
    def grid(self):
        return self._grid

    @cached_property
    def vacuum_toroidal_field(self):
        r0 = float(self._cache.equilibrium.vacuum_toroidal_field.r0)
        b0 = float(self._cache.equilibrium.vacuum_toroidal_field.b0)

        if not r0:
            lim_r = self.wall.limiter.outline.r
            r0 = (min(lim_r)+max(lim_r))*0.5

        if isinstance(self._cache, LazyProxy):
            # logger.debug(self._cache.equilibrium.time_slice.profiles_1d.f)
            b0 = self._cache.equilibrium.time_slice.profiles_1d.f()[-1]/r0

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

    # --------------------------------------------------------------------------

    @cached_property
    def equilibrium(self):
        return Equilibrium(self._cache.equilibrium.time_slice, tokamak=self)

    @property
    def core_profiles(self):
        if self._core_profiles is None:
            self._core_profiles = CoreProfiles(self._cache.core_profiles,
                                               time=self.time,
                                               grid=self.grid,
                                               tokamak=self)
        return self._core_profiles

    @property
    def edge_profiles(self):
        if self._edge_profiles is None:
            self._edge_profiles = EdgeProfiles(self._cache.edge_profiles,
                                               time=self.time,
                                               grid=self.grid,
                                               vacuum_toroidal_field=self.vacuum_toroidal_field)
        return self._edge_profiles

    @cached_property
    def core_transport(self):
        """Core plasma transport of particles, energy, momentum and poloidal flux."""
        return AttributeTree(default_factory_array=lambda _holder=self: CoreTransport(None, grid=_holder.grid, tokamak=_holder))

    @cached_property
    def core_sources(self):
        """Core plasma thermal source terms (for the transport equations of the thermal species).
            Energy terms correspond to the full kinetic energy equation
            (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return AttributeTree(default_factory_array=lambda _holder=self: CoreSources(None, grid=_holder.grid, tokamak=_holder))

    @cached_property
    def edge_transports(self):
        """Edge plasma transport. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return EdgeTransport(self._cache.edge_transport.mode, tokamak=self)

    @cached_property
    def edge_sources(self):
        """Edge plasma sources. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)
        """

        return CoreSources(self._cache.edge_sources.mode, tokamak=self)

    @cached_property
    def transport(self):
        return TransportSolver(self._cache.transport, tokamak=self)

    @cached_property
    def constraints(self):
        return AttributeTree()

    # --------------------------------------------------------------------------
    def update(self, *args,
               time=None,
               core_profiles=None,
               max_iters=1,
               tolerance=0.1,
               ** kwargs):

        convergence = False

        if time is None:
            time = self._time

        if core_profiles is not None:
            core_profiles_prev = CoreProfiles(core_profiles,  time=time, grid=self.grid, tokamak=self)
        elif self._core_profiles is not None:
            core_profiles_prev = self._core_profiles
        else:
            raise RuntimeError(f"Core profiles is not defined!")

        for iter_count in range(max_iters):

            logger.debug(f"Iterator = {iter_count}")

            # try:
            #     profiles = core_profiles_prev.profiles_1d.interpolate(["dpressure_dpsi", "f_df_dpsi"])
            # except Exception:
            #     profiles = None

            # self.equilibrium.update(profiles=profiles, constraints=self.constraints)

            for src in self.core_sources:
                src.update(time=time, equilibrium=self.equilibrium)

            for trans in self.core_transport:
                trans.update(time=time, equilibrium=self.equilibrium)

            core_profiles_next = CoreProfiles(time=time,  grid=self.grid, tokamak=self)

            assert(core_profiles_prev.profiles_1d.grid.rho_tor_norm.shape ==
                   core_profiles_next.profiles_1d.grid.rho_tor_norm.shape)

            self.transport.update(core_profiles_prev,
                                  core_profiles_next,
                                  equilibrium=self.equilibrium,
                                  core_transport=self.core_transport,
                                  core_sources=self.core_sources,
                                  boundary_condition=self.boundary_condition
                                  )

            # .. todo:: inetgrate core and edge
            # edge_profiles_old = copy(edge_profiles_iter)

            # edge_profiles_iter = self._transport_edge_solver(
            #     edge_profiles_old, dt,
            #     core_profiles_next,
            #     equilibrium=self._equilibrium,
            #     transports=self.edge_transports,
            #     sources=self.edge_sources,
            #     **kwargs)

            if self.check_converge(core_profiles_prev, core_profiles_next, tolerance):
                convergence = True

            core_profiles_prev = core_profiles_next

            if convergence:
                break

        if not convergence:
            raise RuntimeError(f"Does not converge! iter_count={iter_count}")
        else:
            self._core_profiles = core_profiles_prev

    def check_converge(self, core_profiles_prev, core_profiles_next, tolerance):
        return True
    # --------------------------------------------------------------------------

    def save(self, uri, *args, **kwargs):
        raise NotImplementedError()

    def plot_machine(self, axis=None, *args, coils=True, wall=True, **kwargs):
        if axis is None:
            axis = plt.gca()
        if wall:
            self.wall.plot(axis, **kwargs.get("wall", {}))
        if coils:
            self.pf_active.plot(axis, **kwargs.get("pf_active", {}))
        axis.axis("scaled")

    def plot(self, axis=None, *args,   **kwargs):

        if axis is None:
            axis = plt.gca()

        self.wall.plot(axis, **kwargs.get("wall", {}))

        self.pf_active.plot(axis, **kwargs.get("pf_active", {}))

        self.equilibrium.plot(axis, **kwargs.get("equilibrium", {}))

        axis.set_aspect('equal')
        axis.axis('scaled')
        # axis.set_xlabel(r"Major radius $R$ [m]")
        # axis.set_ylabel(r"Height $Z$ [m]")
        # axis.legend()

        return axis

    def initialize_profile(self, spec="electrons", n0=1.0e19, w_scale=5,  x_ped=0.9, d_ped=0.2):
        r""" Setup dummy profile　
                core_transport
                core_sources
                core_profiles
        """

        gamma = self.equilibrium.profiles_1d.dvolume_drho_tor  \
            * self.equilibrium.profiles_1d.gm2    \
            / self.equilibrium.profiles_1d.fpol \
            * self.equilibrium.profiles_1d.dpsi_drho_tor \
            / (4.0*(scipy.constants.pi**2))

        j_total = -gamma.derivative  \
            / self.equilibrium.profiles_1d.rho_tor[-1]**2 \
            * self.equilibrium.profiles_1d.dpsi_drho_tor  \
            * (self.equilibrium.profiles_1d.fpol**2) \
            / (scipy.constants.mu_0*self.vacuum_toroidal_field.b0) \
            * (scipy.constants.pi)

        j_total.value[1:] /= self.equilibrium.profiles_1d.dvolume_drho_tor.value[1:]

        j_total.value[0] = 2*j_total.value[1]-j_total.value[2]

        self.core_transport[_next_] = {"identifier": {"name": f"Dummy transport {spec}", "index": 0}}

        self.core_sources[_next_] = {"identifier": {"name": f"Dummy source {spec}", "index": 0}}

        self.core_sources[-1].profiles_1d.j_parallel = j_total.value

        self.core_sources[-1].profiles_1d.conductivity_parallel = 1.0e-8

        if isinstance(spec, str):
            spec = {spec: {}}
        elif not isinstance(spec, collections.abc.Mapping):
            raise TypeError(type(spec))

        rho_tor_norm = self.grid.rho_tor_norm

        rho_tor_boundary = self.grid.rho_tor_boundary

        vpr = Profile(self.equilibrium.profiles_1d.dvolume_drho_tor * rho_tor_boundary,
                      axis=self.equilibrium.profiles_1d.rho_tor_norm)(rho_tor_norm)

        gm3 = Profile(self.equilibrium.profiles_1d.gm3,
                      axis=self.equilibrium.profiles_1d.rho_tor_norm)(rho_tor_norm)

        H = vpr * gm3

        for sp, desc in spec.items():
            n_s = desc.get("n0", n0),
            x_ped_s = desc.get("x_ped", x_ped)
            d_ped_s = desc.get("d_ped", d_ped)
            w_scale_s = desc.get("w_scale", w_scale)

            def n_core(x): return (1-(x/w_scale_s)**2)**2

            def dn_core(x): return -4*x*(1-(x/w_scale_s)**2)/(w_scale_s**2)

            def n_ped(x): return n_core(x_ped_s) - (1.0-x_ped_s) * dn_core(x_ped_s) \
                * (1.0 - np.exp((x-x_ped_s)/(1.0-x_ped_s)))

            def dn_ped(x): return dn_core(x_ped_s) * np.exp((x-x_ped_s)/(1.0-x_ped_s))

            integral_src = -d_ped_s * H * dn_ped(rho_tor_norm)/(rho_tor_boundary**2)

            self.core_transport[-1].profiles_1d[sp].particles.d = \
                lambda x:   2.0 * d_ped_s + (x**2) if x <= x_ped_s else d_ped_s

            self.core_transport[-1].profiles_1d[sp].particles.v = \
                (self.core_transport[-1].profiles_1d[sp].particles.d * dn_core(rho_tor_norm) - d_ped_s*dn_ped(rho_tor_norm)) \
                / (rho_tor_boundary) / n_core(rho_tor_norm) * (rho_tor_norm < x_ped_s)

            self.core_sources[-1].profiles_1d[sp].particles = n_s * integral_src.derivative/vpr

            self.core_profiles.profiles_1d[sp].density = n_s * (n_core(rho_tor_norm)*(rho_tor_norm < x_ped_s) +
                                                               n_ped(rho_tor_norm) * (rho_tor_norm >= x_ped_s))
