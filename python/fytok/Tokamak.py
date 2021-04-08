
import collections
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.AttributeTree import AttributeTree
from spdm.data.List import List
from spdm.data.Node import Node, _next_
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.numerical.Function import Function
from spdm.util.logger import logger

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
from .modules.utilities.RadialGrid import RadialGrid


class Tokamak(PhysicalGraph):
    """Tokamak
        功能：
            - 描述装置在单一时刻的状态，
            - 在时间推进时，确定各个子系统之间的依赖和演化关系，

    """

    def __init__(self, *args, radial_grid=None,  **kwargs):
        super().__init__(*args,  **kwargs)
        self._equilibrium = None
        self._core_profiles = None
        self._edge_profiles = None
        self._time = 0.0
        self._radial_grid = radial_grid or self["grid"]

    @property
    def radial_grid(self):
        if isinstance(self._radial_grid, collections.abc.Mapping):
            self._radial_grid = RadialGrid(self._radial_grid["axis"],
                                           label=self._radial_grid["label"],
                                           equilibrium=self.equilibrium)
        elif self._radial_grid == None:
            self._radial_grid = RadialGrid(equilibrium=self.equilibrium)

        return self._radial_grid

    @property
    def time(self):
        return self._time

    @property
    def vacuum_toroidal_field(self):
        return self._vacuum_toroidal_field

    @cached_property
    def wall(self):
        return Wall(self["wall.description_2d"], parent=self)

    @cached_property
    def tf(self):
        return TF(self["tf"], parent=self)

    @cached_property
    def pf_active(self):
        return PFActive(self["pf_active"], parent=self)

    # --------------------------------------------------------------------------
    @cached_property
    def constraints(self):
        return PhysicalGraph(self["constraints"], parent=self)

    @property
    def equilibrium(self) -> Equilibrium:
        if self._equilibrium is None:
            if self["equilibrium.time_slice"] != None:
                eq = self["equilibrium.time_slice"]
            else:
                eq = self["equilibrium"]

            self._equilibrium = Equilibrium(eq,
                                            time=self.time,
                                            vacuum_toroidal_field=self.vacuum_toroidal_field,
                                            constraints=self.constraints,
                                            wall=self.wall,
                                            pf_active=self.pf_active,
                                            tf=self.tf,
                                            parent=self)
        return self._equilibrium

    @property
    def core_profiles(self):
        if self._core_profiles is None:
            if self["core_profiles.profiles_1d"] != None:
                core_profiles = self["core_profiles.profiles_1d"]
            else:
                core_profiles = self["core_profiles"]

            self._core_profiles = CoreProfiles(self.radial_grid, core_profiles,  time=self.time, parent=self)

        return self._core_profiles

    @cached_property
    def core_transport(self):
        """Core plasma transport of particles, energy, momentum and poloidal flux."""
        return List(self["core_transport"],
                    default_factory=lambda d, *args, **kwargs: CoreTransport(self.radial_grid, d, time=self.time, *args, **kwargs), parent=self)

    @cached_property
    def core_sources(self):
        """Core plasma thermal source terms (for the transport equations of the thermal species).
            Energy terms correspond to the full kinetic energy equation
            (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return List(self["core_sources"],
                    default_factory=lambda d, *args, **kwargs: CoreSources(self.radial_grid, d, time=self.time, *args, **kwargs), parent=self)

    @cached_property
    def edge_profiles(self):
        return EdgeProfiles(self["edge_profiles"], parent=self)

    @cached_property
    def edge_transport(self):
        """Edge plasma transport. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return EdgeTransport(self["edge_transport.mode"], parent=self)

    @cached_property
    def edge_sources(self):
        """Edge plasma sources. Energy terms correspond to the full kinetic energy equation
         (i.e. the energy flux takes into account the energy transported by the particle flux)
        """
        return EdgeSources(self["edge_sources.mode"], parent=self)

    @cached_property
    def transport_solver(self):
        return TransportSolver(self["transport_solver"],
                               time=self.time,
                               vacuum_toroidal_field=self.vacuum_toroidal_field,
                               equilibrium=self.equilibrium,
                               core_transport=self.core_transport,
                               edge_transport=self.edge_transport,
                               core_sources=self.core_sources,
                               edge_sources=self.edge_sources,
                               parent=self)

    def initialize(self, spec=None, npoints=128):
        r"""
            Setup dummy profile　
                core_transport
                core_sources
                core_profiles
        """

        if not isinstance(spec, AttributeTree):
            spec = AttributeTree(spec)

        # gamma = self.equilibrium.magnetic_flux_coordinates.dvolume_drho_tor  \
        #     * self.equilibrium.magnetic_flux_coordinates.gm2    \
        #     / self.equilibrium.magnetic_flux_coordinates.fpol \
        #     * self.equilibrium.magnetic_flux_coordinates.dpsi_drho_tor \
        #     / (4.0*(scipy.constants.pi**2))
        # gamma = Function(rho, gamma)
        # j_total = -gamma.derivative  \
        #     / self.equilibrium.magnetic_flux_coordinates.rho_tor[-1]**2 \
        #     * self.equilibrium.magnetic_flux_coordinates.dpsi_drho_tor  \
        #     * (self.equilibrium.magnetic_flux_coordinates.fpol**2) \
        #     / (scipy.constants.mu_0*self.vacuum_toroidal_field.b0) \
        #     * (scipy.constants.pi)
        # j_total[1:] /= self.equilibrium.magnetic_flux_coordinates.dvolume_drho_tor[1:]
        # j_total[0] = 2*j_total[1]-j_total[2]

        r_ped = spec.r_ped
        # rho_core = np.linspace(0.0, r_ped, npoints, endpoint=False)

        # rho_edge = np.linspace(r_ped, 1.0, int((1.0-r_ped)*npoints))

        # rho = np.hstack([rho_core, rho_edge])

        rho_n = np.linspace(0.0, 1.0, npoints)

        self._radial_grid = {"axis": rho_n, "label": "rho_tor_norm"}

        p_src = spec.electron.density.source

        D_diff = spec.electron.density.diffusivity

        v_pinch = spec.electron.density.pinch

        # if isinstance(spec.electron.density.diffusivity, Function):

        # elif isinstance(spec.electron.density.diffusivity, AttributeTree):
        #     D_diff = Function(rho_n,  spec.electron.density.diffusivity[0], spec.electron.density.diffusivity[1])
        # elif callable(spec.electron.density.diffusivity):
        #     D_diff = Function(rho_n, spec.electron.density.diffusivity)
        # else:
        #     raise NotImplementedError(type(spec.electron.density.diffusivity))
        # D0 = spec.electron.density.diffusivity.D0
        # D1 = spec.electron.density.diffusivity.D1
        # D2 = spec.electron.density.diffusivity.D2
        # D_diff = Function(rho_n, [lambda r:r < r_ped, lambda r:r >= r_ped],
        #                   [lambda x:D0 + D1 * (x**2), lambda x: D2])
        # v_pinch = -D_diff*rho_n * spec.electron.density.pinch_number.V0 /   self.equilibrium.vacuum_toroidal_field.r0
        # def n_core(x): return (1-x**4)**2
        # def dn_core(x): return -4*x*(1-x**2)
        # def n_ped(x, r_ped=r_ped): return n_core(r_ped) - (1.0-r_ped)/2.0 * \
        #     dn_core(r_ped) * (1.0 - np.exp(2.0*(x-r_ped)/(1.0-r_ped)))
        # #     def dn_ped(x): return dn_core(x_ped) * np.exp((x-x_ped)/(1.0-x_ped))

        self.core_profiles.electrons.density = spec.electron.density.n0

        # Function(rho_n, spec.electron.density.n0 * (1-rho_n**4)**2)

        self.core_transport[_next_] = {
            "electrons": {
                "particles": {
                    "d": D_diff,
                    "v": v_pinch,
                }
            }
        }

        self.core_sources[_next_] = {
            "electrons": {
                "particles": p_src
            },
            # "j_parallel": j_total, "conductivity_parallel": 1.0e-8
        }

        # self.core_sources[-1]["profiles_1d.j_parallel"] = j_total
        # self.core_sources[-1]["profiles_1d.conductivity_parallel"] = 1.0e-8
        # rho = self.grid.rho

        # rho_tor_boundary = self.equilibrium.profiles_1d.rho_tor[-1]

        # vpr = Function(self.equilibrium.profiles_1d.rho_tor_norm,
        #                self.equilibrium.profiles_1d.dvolume_drho_tor)

        # gm3 = Function(self.equilibrium.profiles_1d.rho_tor_norm,
        #                self.equilibrium.profiles_1d.gm3)

        # H = vpr * gm3

        # for sp, desc in spec.items():
        #     n_s = desc.get("density", n0)
        #     w_scale_s = desc.get("w_scale", w_scale)
        #     def n_core(x): return (1-(x/w_scale_s)**2)**2
        #     def dn_core(x): return -4*x*(1-(x/w_scale_s)**2)/(w_scale_s**2)
        #     def n_ped(x): return n_core(x_ped) - (1.0-x_ped) * dn_core(x_ped) * (1.0 - np.exp((x-x_ped)/(1.0-x_ped)))
        #     def dn_ped(x): return dn_core(x_ped) * np.exp((x-x_ped)/(1.0-x_ped))
        #     integral_src = Function(rho, -d_ped * H * dn_ped(rho)/(rho_tor_boundary**2))
        #     self.core_transport.profiles_1d[sp].particles.d = lambda x: 2.0 * d_ped + (x**2)
        #     self.core_transport.profiles_1d[sp].particles.v = (self.core_transport.profiles_1d[sp].particles.d(rho) * dn_core(rho) - d_ped*dn_ped(rho)) \
        #         / (rho_tor_boundary) / n_core(rho) * (rho < x_ped)
        #     self.core_sources.profiles_1d[sp].particles = n_s * integral_src.derivative/vpr
        #     desc["density"] = n_s * (n_core(rho)*(rho < x_ped) +
        #                              n_ped(rho) * (rho >= x_ped))
        #     self.core_profiles.profiles_1d[sp] |= desc
        #     logger.debug(self.core_sources)
        # if "electrons" not in spec:
        #     raise NotImplementedError()

    def update(self, time=None, tolerance=1.0e-6, max_step=1,  **kwargs):
        if time is not None:
            self._time = time

        core_profiles_prev = self.core_profiles

        for nstep in range(max_step):
            logger.debug(f"time={self.time}  iterator step {nstep}/{max_step}")

            if kwargs.get("equilibrium", False) is not False:
                self.equilibrium.update(time=self.time, core_profiles=self.core_profiles,
                                        **kwargs.get("equilibrium", {}))

            if kwargs.get("core_transport", False) is not False:
                self.core_transport.update(time=self.time, **kwargs.get("core_transport", {}))

            if kwargs.get("core_sources", False) is not False:
                self.core_sources.update(time=self.time, **kwargs.get("core_sources", {}))

            if kwargs.get("edge_transport", False) is not False:
                self.edge_transport.update(time=self.time, **kwargs.get("edge_transport", {}))

            if kwargs.get("edge_sources", False) is not False:
                self.edge_sources.update(time=self.time, **kwargs.get("edge_sources", {}))

            core_profiles_next = self.transport_solver.solve(core_profiles_prev, **kwargs.get("transport_solver", {}))

        #    if core_profiles_next.conv(core_profiles_prev) < tolerance:
        #         break
            core_profiles_prev = core_profiles_next

        self._core_profiles = core_profiles_next

    def plot(self, axis=None, *args,   **kwargs):

        if axis is None:
            axis = plt.gca()

        if kwargs.get("wall", True) is not False:
            self.wall.plot(axis, **kwargs.get("wall", {}))

        if kwargs.get("pf_active", True) is not False:
            self.pf_active.plot(axis, **kwargs.get("pf_active", {}))

        if kwargs.get("equilibrium", True) is not False:
            self.equilibrium.plot(axis, **kwargs.get("equilibrium", {}))

        axis.set_aspect('equal')
        axis.axis('scaled')
        axis.set_xlabel(r"Major radius $R$ [m]")
        axis.set_ylabel(r"Height $Z$ [m]")
        # axis.legend()
        return axis
