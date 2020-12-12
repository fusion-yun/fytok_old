import collections
import copy
import math
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from spdm.util.AttributeTree import AttributeTree, _last_, _next_
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.Profiles import Profile

from fytok.CoreProfiles import CoreProfiles
from fytok.CoreSources import CoreSources
from fytok.CoreTransport import CoreTransport
from fytok.EdgeProfiles import EdgeProfiles
from fytok.EdgeSources import EdgeSources
from fytok.EdgeTransport import EdgeTransport
from fytok.Equilibrium import Equilibrium
from fytok.PFActive import PFActive
from fytok.RadialGrid import RadialGrid
from fytok.TF import TF
from fytok.TransportSolver import TransportSolver
from fytok.Wall import Wall


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
        self._core_profiles = None
        if rho_tor_norm is None:
            self._rho_tor_norm = np.linspace(0, 1.0, 129)
        else:
            self._rho_tor_norm = rho_tor_norm

        self._edge_profiles = None

    # --------------------------------------------------------------------------
    @property
    def time(self):
        return self._time

    @property
    def grid(self):
        # logger.debug(self.equilibrium.profiles_1d.rho_tor_norm)
        # return RadialGrid(self.equilibrium.profiles_1d.rho_tor_norm, equilibrium=self.equilibrium)
        return RadialGrid(self._rho_tor_norm, equilibrium=self.equilibrium)

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

    def add_dummy_profile(self, spec="electrons", rho_ped=0.95, n0=1.0e19, D_bdry=0.2):
        r""" Setup dummy porfilse　
                core_transport
                core_sources
                core_profiles
        """
        if isinstance(spec, list):
            spec = [spec]

        x = self.grid.rho_tor_norm

        rho_bdry=rho_ped
        
        rho_src_bdry = 1.0-4*(1.0-rho_ped)

        self.core_transport[_next_] = {"identifier": {"name": f"Dummy transport {spec}", "index": 0}}
        self.core_sources[_next_] = {"identifier": {"name": f"Dummy source {spec}", "index": 0}}

        trans = self.core_transport[-1].profiles_1d
        sources = self.core_sources[-1].profiles_1d

        trans.conductivity_parallel = 1.0e-8

        # V=np.piecewise( x, [x < rho_ped,x > rho_ped], [lambda x:-(x**3)  , 0 ])

        # S_pel = Profile(scipy.stats.norm.pdf((x-0.7)/0.1)*np.sqrt(scipy.constants.pi*2.0), axis=x)

        # rho_bdry=0.95
        # rho_src_bdry = 0.80
        D_bdry=0.2

        def s_edge(x):return 100*((x-rho_src_bdry)/(1.0-rho_src_bdry))**2

        S_edge=Profile(np.piecewise(x, [x<rho_src_bdry, x >=rho_src_bdry],[0,s_edge]), axis=x)

        def int_s_edge(x ):return scipy.integrate.quad(s_edge,rho_src_bdry,x)[0]

        int_S_edge=Profile(np.piecewise(x, [x<rho_src_bdry, x >= rho_src_bdry], [0, lambda r:np.array([ int_s_edge(s) for s in r])]), axis=x)

        def n_ped_prime(x): return scipy.integrate.quad(s_edge,rho_src_bdry,x )[0]

        n_ped_bdr=((1-(rho_bdry/3)**2)**2) 

        def n_core(x,  w=3): return ((1-(x/w)**2)**2) if x<rho_bdry else n_ped_bdr

        def n_ped(s):return -scipy.integrate.quad(n_ped_prime,rho_src_bdry,s )[0] /D_bdry/n_ped_bdr if (s>rho_src_bdry) else 0  

        self.core_profiles.profiles_1d.electrons.density = lambda s: n0 * n_core(s) * (1.0 + n_ped(s)) 

        sources[spec].particles = Profile(np.piecewise(x, [x < rho_ped, x >= rho_ped], [0, s_edge]), axis=x)*n0

        D = np.piecewise(x, [x < rho_ped, x > rho_ped], [lambda x:2.0 * D_bdry + (x**2), D_bdry])

        trans[spec].particles.d = D

        trans[spec].particles.d._interpolator = scipy.interpolate.interp1d

        # trans[spec].particles.v = (trans[spec].particles.d
        #                            * self.core_profiles.profiles_1d[spec].density.derivative
        #                            + int_S_edge)\
        #     / self.core_profiles.profiles_1d[spec].density

        trans[spec].particles.v._interpolator = scipy.interpolate.interp1d

        # gamma = tok.equilibrium.profiles_1d.dvolume_drho_tor  \
        #     * tok.equilibrium.profiles_1d.gm2    \
        #     / tok.equilibrium.profiles_1d.fpol \
        #     * tok.equilibrium.profiles_1d.dpsi_drho_tor \
        #     / (4.0*(constants.pi**2))

        # j_total = - gamma.derivative \
        #     / tok.equilibrium.profiles_1d.rho_tor[-1]**2 \
        #     * tok.equilibrium.profiles_1d.dpsi_drho_tor  \
        #     * (tok.equilibrium.profiles_1d.fpol**2) \
        #     / (constants.mu_0*tok.vacuum_toroidal_field.b0) \
        #     * (constants.pi)

        # j_total[1:] /= tok.equilibrium.profiles_1d.dvolume_drho_tor[1:]

        # j_total[0] = 2*j_total[1]-j_total[2]

        # sources.j_parallel = Profile(j_total, tok.equilibrium.profiles_1d.rho_tor_norm)
