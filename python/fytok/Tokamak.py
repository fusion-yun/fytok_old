import typing
from copy import copy

from spdm.data.Entry import open_entry
from spdm.data.sp_property import SpDict, sp_property
from spdm.geometry.GeoObject import GeoObject
from spdm.utils.logger import logger

from ._imas.lastest import __version__ as imas_version

# ---------------------------------
from .modules.CoreProfiles import CoreProfiles
from .modules.CoreSources import CoreSources
from .modules.CoreTransport import CoreTransport

# ---------------------------------
# from .modules.EdgeProfiles import EdgeProfiles
# from .modules.EdgeSources import EdgeSources
# from .modules.EdgeTransport import EdgeTransport
# from .modules.EdgeTransportSolver import EdgeTransportSolver
# ---------------------------------
from .modules.Equilibrium import Equilibrium

# ---------------------------------
from .modules.Magnetics import Magnetics
from .modules.PFActive import PFActive
from .modules.TF import TF
from .modules.TransportSolverNumerics import TransportSolverNumerics
from .modules.Wall import Wall


class Tokamak(SpDict):
    """Tokamak
    功能：
        - 描述装置在单一时刻的状态，
        - 在时间推进时，确定各个子系统之间的依赖和演化关系，
    """

    def __init__(self, data: str | typing.Dict = None, **kwargs):
        imas_version_major, *_ = imas_version.split(
            "."
        )  # imas_version_minor, imas_version_patch,

        if not isinstance(data, dict):
            entry = open_entry(data, schema=f"imas/{imas_version_major}")
            data = None
        else:
            entry = None

        super().__init__(data, entry=entry, **kwargs)

    time: float = sp_property(default_value=0.0)

    name: str = sp_property(default_value="unknown")

    description: str = sp_property(default_value="empty tokamak")

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

    def advance(
        self, *args, dt=None, do_refresh=False, **kwargs
    ) -> CoreProfiles.Profiles1d:
        self._time += dt

        core_profiles_1d_prev = self.core_profiles.profiles_1d.current

        equilibrium = self.equilibrium.advance(
            time=self.time,
            core_profile_1d=core_profiles_1d_prev,
            wall=self.wall,
            pf_active=self.pf_active,
        )

        core_transport_profiles_1d = self.core_transport.advance(
            time=self.time,
            equilibrium=equilibrium,
            core_profile_1d=core_profiles_1d_prev,
        )

        core_source_profiles_1d = self.core_sources.advance(
            time=self.time,
            equilibrium=equilibrium,
            core_profile_1d=core_profiles_1d_prev,
        )

        core_profiles_1d_next = self.transport_solver.solve(
            equilibrium=equilibrium,
            core_profile_1d=core_profiles_1d_prev,
            core_transport_profiles_1d=core_transport_profiles_1d,
            core_source_profiles_1d=core_source_profiles_1d,
        )

        self.core_profiles.advance(core_profiles_1d_next)

        self.core_profiles.advance(core_profiles_1d_next)
        self.core_sources.advance()
        self.core_transport.advance()

        if do_refresh:
            return self.refresh()
        else:
            return core_profiles_1d_next

    def refresh(self, *args, tolerance=1.0e-4, max_iteration=1, **kwargs):
        self.equilibrium.refresh(
            core_profiles_1d=core_profiles_1d_iter,
            wall=self.wall,
            pf_active=self.pf_active,
            tf=self.tf,
            tolerance=tolerance,
            **kwargs,
        )

        residual = tolerance

        core_profiles_1d_iter = copy(self.core_profiles.profiles_1d.current)

        for step_num in range(max_iteration):
            equilibrium_time_slice = self.equilibrium.time_slice.current

            self.core_transport.refresh(
                equilibrium=equilibrium_time_slice,
                core_profile_1d=core_profiles_1d_iter,
            )

            core_transport_profiles_1d = (
                self.core_transport.model.combined.profiles_1d.current
            )

            self.core_sources.refresh(
                equilibrium=equilibrium_time_slice,
                core_profile_1d=core_profiles_1d_iter,
            )

            core_source_profiles_1d = (
                self.core_sources.source.combined.profiles_1d.current
            )

            core_profiles_1d_next = self.transport_solver.solve(
                equilibrium=equilibrium_time_slice,
                core_profiles_prev=core_profiles_1d_iter,
                core_transport_profiles_1d=core_transport_profiles_1d,
                core_source_profiles_1d=core_source_profiles_1d,
            )

            residual = self.check_converge(core_profiles_1d_iter, core_profiles_1d_next)

            if residual <= tolerance:
                break
            else:
                core_profiles_1d_iter = core_profiles_1d_next
        else:
            logger.debug(
                f"time={self.time}  iterator step {step_num}/{max_iteration} residual={residual}"
            )

        if residual >= tolerance:
            logger.warning(
                f"The solution does not converge, and the number of iterations exceeds the maximum {max_iteration}"
            )

        return core_profiles_1d_iter

    @property
    def __geometry__(self) -> GeoObject:
        geo = {
            "wall": self.wall.__geometry__,
            "pf_active": self.pf_active.__geometry__,
            "magnetics": self.magnetics.__geometry__,
            "equilibrium": self.equilibrium.__geometry__,
        }

        return geo, {
            "xlabel": r"Major radius $R$ [m]",
            "ylabel": r"Height $Z$ [m]",
        }

    # def plot(self, axis=None, /,  **kwargs):

    #     import matplotlib.pylab as plt

    #     if axis is None:
    #         axis = plt.gca()

    #     if kwargs.get("wall", True) is not False:
    #         self.wall.plot(axis, **kwargs.get("wall", {}))

    #     if kwargs.get("pf_active", True) is not False:
    #         self.pf_active.plot(axis, **kwargs.get("pf_active", {}))

    #     if kwargs.get("magnetics", True) is not False:
    #         self.magnetics.plot(axis,  **kwargs.get("magnetics", {}))

    #     if kwargs.get("equilibrium", True) is not False:
    #         self.equilibrium.plot(axis,  **kwargs.get("equilibrium", {}))

    #     axis.set_aspect('equal')
    #     axis.axis('scaled')
    #     axis.set_xlabel(r"Major radius $R$ [m]")
    #     axis.set_ylabel(r"Height $Z$ [m]")

    #     # axis.legend()

    #     return axis

    # def display(self, *args, **kwargs):
    #     return display([(self.wall, kwargs.pop("wall", {})),
    #                     (self.pf_active, kwargs.pop("pf_active", {})),
    #                     (self.magnetics, kwargs.pop("magnetics", {})),
    #                     (self.equilibrium, kwargs.pop("equilibrium", {})),
    #                     ], *args,
    #                    xlabel=kwargs.pop("xlabel", r"Major radius $R$ [m]"),
    #                    ylabel=kwargs.pop("ylabel", r"Height $Z$ [m]"),
    #                    title=kwargs.pop("title", f"{self.name} time={self.time}s"),
    #                    **kwargs)

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
