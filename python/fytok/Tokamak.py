
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy, copy
from spdm.utils.logger import logger
from spdm.data.sp_property import sp_property, SpDict
from spdm.data.Dict import Dict
from spdm.data.Node import Node
# ---------------------------------
from .modules.Magnetics import Magnetics
from .modules.PFActive import PFActive
from .modules.TF import TF
from .modules.Wall import Wall
# ---------------------------------
from .modules.CoreProfiles import CoreProfiles
from .modules.CoreSources import CoreSources
from .modules.CoreTransport import CoreTransport
from .modules.TransportSolverNumerics import TransportSolverNumerics
# ---------------------------------
# from .modules.EdgeProfiles import EdgeProfiles
# from .modules.EdgeSources import EdgeSources
# from .modules.EdgeTransport import EdgeTransport
# from .modules.EdgeTransportSolver import EdgeTransportSolver
# ---------------------------------
from .modules.Equilibrium import Equilibrium


class Tokamak(SpDict):
    """Tokamak
        功能：
            - 描述装置在单一时刻的状态，
            - 在时间推进时，确定各个子系统之间的依赖和演化关系，
    """

    def __init__(self, *args,  **kwargs):
        super().__init__(*args,  **kwargs)
        self._time = 0.0

    @property
    def time(self) -> float: return self._time

    wall: Wall = sp_property()

    tf: TF = sp_property()

    pf_active: PFActive = sp_property()

    magnetics: Magnetics = sp_property()

    equilibrium: Equilibrium = sp_property()

    core_profiles: CoreProfiles = sp_property()

    core_transport: CoreTransport = sp_property()

    core_sources: CoreSources = sp_property()

    # edge_profiles: EdgeProfiles = sp_property()

    # edge_transport: EdgeTransport = sp_property()

    # edge_sources: EdgeSources = sp_property()

    # edge_transport_solver: EdgeTransportSolver = sp_property()

    transport_solver: TransportSolverNumerics = sp_property()

    def check_converge(self, *args, **kwargs) -> float:
        return 0.0

    def advance(self, *args, dt=None, do_update=False, **kwargs) -> CoreProfiles.Profiles1d:
        self._time += dt

        core_profiles_1d_prev = self.core_profiles.profile_1d.current
        
        equilibrium = self.equilibrium.advance(
            time=self.time,
            core_profile_1d=core_profiles_1d_prev,
            wall=self.wall,
            pf_active=self.pf_active)

        core_transport_profiles_1d = self.core_transport.advance(
            time=self.time,
            equilibrium=equilibrium,
            core_profile_1d=core_profiles_1d_prev)

        core_source_profiles_1d = self.core_sources.advance(
            time=self.time,
            equilibrium=equilibrium,
            core_profile_1d=core_profiles_1d_prev)

        core_profiles_1d_next = self.transport_solver.solve(
            equilibrium=equilibrium,
            core_profile_1d=core_profiles_1d_prev,
            core_transport_profiles_1d=core_transport_profiles_1d,
            core_source_profiles_1d=core_source_profiles_1d)

        self.core_profiles.advance(core_profiles_1d_next)

        if do_update:
            return self.update()
        else:
            return core_profiles_1d_next

    def update(self, tolerance=1.0e-4, max_iteration=1) -> CoreProfiles.Profiles1d:

        residual = tolerance

        core_profiles_1d_iter = deepcopy(self.core_profiles.profiles_1d.current)

        for step_num in range(max_iteration):
            equilibrium = self.equilibrium.update(
                core_profile_1d=core_profiles_1d_iter,
                wall=self.wall,
                pf_active=self.pf_active)

            core_transport_profiles_1d = self.core_transport.update(
                equilibrium=equilibrium,
                core_profile_1d=core_profiles_1d_iter)

            core_source_profiles_1d, *_ = self.core_sources.advance(
                equilibrium=equilibrium,
                core_profile_1d=core_profiles_1d_iter)

            core_profiles_1d_next = self.transport_solver.solve(
                equilibrium=equilibrium,
                core_profiles_prev=core_profiles_1d_iter,
                core_transport_profiles_1d=core_transport_profiles_1d,
                core_source_profiles_1d=core_source_profiles_1d)

            residual = self.check_converge(core_profiles_1d_iter,  core_profiles_1d_next)

            if residual <= tolerance:
                break
            else:
                core_profiles_1d_iter = core_profiles_1d_next
        else:
            logger.debug(f"time={self.time}  iterator step {step_num}/{max_iteration} residual={residual}")

        self.core_profiles.update(core_profiles_1d_iter)

        if residual >= tolerance:
            logger.warning(
                f"The solution does not converge, and the number of iterations exceeds the maximum {max_iteration}")

        return core_profiles_1d_iter

    def plot(self, axis=None, /,  **kwargs):

        if axis is None:
            axis = plt.gca()

        if kwargs.get("wall", True) is not False:
            self.wall.plot(axis, **kwargs.get("wall", {}))

        if kwargs.get("pf_active", True) is not False:
            self.pf_active.plot(axis, **kwargs.get("pf_active", {}))

        if kwargs.get("magnetics", True) is not False:
            self.magnetics.plot(axis,  **kwargs.get("magnetics", {}))

        if kwargs.get("equilibrium", True) is not False:
            self.equilibrium.plot(axis,  **kwargs.get("equilibrium", {}))
        axis.set_aspect('equal')
        axis.axis('scaled')
        axis.set_xlabel(r"Major radius $R$ [m]")
        axis.set_ylabel(r"Height $Z$ [m]")
        # axis.legend()

        return axis

    # def initialize(self):
    #     r"""
    #         Set initial conditions self-consistently

    #     """

    #     gamma = self.equilibrium.profiles_1d.dvolume_drho_tor  \
    #         * self.equilibrium.profiles_1d.gm2    \
    #         / self.equilibrium.profiles_1d.fpol \
    #         * self.equilibrium.profiles_1d.dpsi_drho_tor \
    #         / (TWOPI**2)

    #     j_total = -gamma.derivative  \
    #         / self.equilibrium.profiles_1d.rho_tor[-1]**2 \
    #         * self.equilibrium.profiles_1d.dpsi_drho_tor  \
    #         * (self.equilibrium.profiles_1d.fpol**2) \
    #         / (constants.mu_0*self.vacuum_toroidal_field.b0) \
    #         * (constants.pi)

    #     j_total[1:] /= self.equilibrium.profiles_1d.dvolume_drho_tor[1:]
    #     j_total[0] = 2*j_total[1]-j_total[2]

    #     self.core_sources["j_parallel"] = j_total
