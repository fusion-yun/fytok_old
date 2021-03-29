import collections
from functools import cached_property

import numpy as np
import scipy
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Function import Function
from spdm.data.Node import Node, _next_
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.util.logger import logger

from .modules.operation.PulseSchedule import PulseSchedule
from .modules.transport.TransportSolver import TransportSolver
from .Tokamak import Tokamak


class Scenario(PhysicalGraph):
    """
        Scenario

    """

    def __init__(self,  *args,  **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property    
    def tokamak(self):
        return Tokamak(self["tokamak"], parent=self)

    @cached_property
    def pulse_schedule(self):
        return PulseSchedule(self["pluse_schedule"], parent=self)

    @cached_property
    def transport_solver(self):
        return TransportSolver(self["transport_solver"], parent=self)

    # --------------------------------------------------------------------------

    def update(self, *args, time=None,   max_iters=1,  tolerance=0.1,   ** kwargs):

        convergence = False

        if time is None:
            time = self._time

        core_profiles_prev = self.core_profiles

        for iter_count in range(max_iters):
            logger.debug(f"Iterator = {iter_count}")

            for src in self.core_sources:
                src.update(time=time, equilibrium=self.equilibrium)

            for trans in self.core_transport:
                trans.update(time=time, equilibrium=self.equilibrium)

            core_profiles_next = self.transport.update(core_profiles_prev,
                                                       equilibrium=self.equilibrium,
                                                       core_transport=self.core_transport,
                                                       core_sources=self.core_sources,
                                                       boundary_condition=self.boundary_condition
                                                       )

            # .. todo:: integrate core and edge
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
                break

            core_profiles_prev = core_profiles_next

            self.equilibrium.update(time=time, profiles=core_profiles_next, constraints=self.constraints)

        if not convergence:
            raise RuntimeError(f"Does not converge! iter_count={iter_count}")
        else:
            self._core_profiles = core_profiles_next

    def check_converge(self, core_profiles_prev, core_profiles_next, tolerance):
        return True

    def initialize(self, spec=None, npoints=128):
        r"""
            Setup dummy profile　
                core_transport
                core_sources
                core_profiles
        """

        conf = AttributeTree(spec)

        pedestal_top = conf.pedestal_top or 0.88

        rho_tor_norm_core = np.linspace(0.0, pedestal_top, npoints, endpoint=False)
        rho_tor_norm_edge = np.linspace(pedestal_top, 1.0, int((1.0-pedestal_top)*npoints))
        # logger.debug((rho_tor_norm_core, rho_tor_norm_edge))

        rho_tor_norm = np.hstack([rho_tor_norm_core, rho_tor_norm_edge])

        logger.debug(rho_tor_norm.shape)

        p_src = Function(rho_tor_norm, conf.particle.source.S0 * conf.particle.source.profile(rho_tor_norm))

        logger.debug(conf.particle_diffusivity.D0 )

        p_diff = Function(rho_tor_norm,
                          np.hstack([conf.particle_diffusivity.D0 + conf.particle_diffusivity.D1 * rho_tor_norm_core**2,
                                     np.full(len(rho_tor_norm_edge), conf.particle_diffusivity.D0)]))

        p_pinch = Function(rho_tor_norm, p_diff*(rho_tor_norm**2) *
                           conf.particle.pinch_number.V0/self.tokamak.vacuum_toroidal_field.r0)

        # gamma = self.tokamak.equilibrium.magnetic_flux_coordinates.dvolume_drho_tor  \
        #     * self.tokamak.equilibrium.magnetic_flux_coordinates.gm2    \
        #     / self.tokamak.equilibrium.magnetic_flux_coordinates.fpol \
        #     * self.tokamak.equilibrium.magnetic_flux_coordinates.dpsi_drho_tor \
        #     / (4.0*(scipy.constants.pi**2))
        # gamma = Function(rho_tor_norm, gamma)
        # j_total = -gamma.derivative  \
        #     / self.tokamak.equilibrium.magnetic_flux_coordinates.rho_tor[-1]**2 \
        #     * self.tokamak.equilibrium.magnetic_flux_coordinates.dpsi_drho_tor  \
        #     * (self.tokamak.equilibrium.magnetic_flux_coordinates.fpol**2) \
        #     / (scipy.constants.mu_0*self.vacuum_toroidal_field.b0) \
        #     * (scipy.constants.pi)
        # j_total[1:] /= self.tokamak.equilibrium.magnetic_flux_coordinates.dvolume_drho_tor[1:]
        # j_total[0] = 2*j_total[1]-j_total[2]

        j_total = None

        self.tokamak.core_transport = {"identifier": {"name": f"Dummy transport", "index": 0}}

        self.tokamak.core_sources = {"identifier": {"name": f"Dummy source", "index": 0},
                                     "profiles_1d": {"j_parallel": j_total, "conductivity_parallel": 1.0e-8}}

        # self.core_sources[-1]["profiles_1d.j_parallel"] = j_total
        # self.core_sources[-1]["profiles_1d.conductivity_parallel"] = 1.0e-8
        # rho_tor_norm = self.grid.rho_tor_norm

        rho_tor_boundary = self.tokamak.equilibrium.magnetic_flux_coordinates.rho_tor[-1]

        psi_norm = self.tokamak.equilibrium.magnetic_flux_coordinates.rho_tor_norm.invert(rho_tor_norm)

        vpr = self.tokamak.equilibrium.magnetic_flux_coordinates.dvolume_drho_tor(psi_norm)

        gm3 = self.tokamak.equilibrium.magnetic_flux_coordinates.gm3(psi_norm)

        H = vpr * gm3

        for sp, desc in spec.items():
            n_s = desc.get("density", n0)

            w_scale_s = desc.get("w_scale", w_scale)

            def n_core(x): return (1-(x/w_scale_s)**2)**2

            def dn_core(x): return -4*x*(1-(x/w_scale_s)**2)/(w_scale_s**2)

            def n_ped(x): return n_core(x_ped) - (1.0-x_ped) * dn_core(x_ped) * (1.0 - np.exp((x-x_ped)/(1.0-x_ped)))

            def dn_ped(x): return dn_core(x_ped) * np.exp((x-x_ped)/(1.0-x_ped))

            integral_src = Function(rho_tor_norm, -d_ped * H * dn_ped(rho_tor_norm)/(rho_tor_boundary**2))

            self.tokamak.core_transport.profiles_1d[sp].particles.d = lambda x: 2.0 * d_ped + (x**2)

            self.tokamak.core_transport.profiles_1d[sp].particles.v = (self.core_transport.profiles_1d[sp].particles.d(rho_tor_norm) * dn_core(rho_tor_norm) - d_ped*dn_ped(rho_tor_norm)) \
                / (rho_tor_boundary) / n_core(rho_tor_norm) * (rho_tor_norm < x_ped)

            self.tokamak.core_sources.profiles_1d[sp].particles = n_s * integral_src.derivative/vpr

            desc["density"] = n_s * (n_core(rho_tor_norm)*(rho_tor_norm < x_ped) +
                                     n_ped(rho_tor_norm) * (rho_tor_norm >= x_ped))

            self.core_profiles.profiles_1d[sp] |= desc

            logger.debug(self.core_sources)

        if "electrons" not in spec:
            raise NotImplementedError()
