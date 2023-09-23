import typing
from copy import copy

from fytok.utils.logger import logger
from spdm.data.Entry import open_entry
from spdm.data.sp_property import SpTree, sp_property
from spdm.data.HTree import HTree
from spdm.geometry.GeoObject import GeoObject
from spdm.utils.tags import _not_found_
from spdm.utils.tree_utils import merge_tree_recursive

from ._imas.lastest import __version__ as imas_version
# ---------------------------------
from .modules.CoreProfiles import CoreProfiles
from .modules.CoreSources import CoreSources
from .modules.CoreTransport import CoreTransport
from .modules.Equilibrium import Equilibrium

# from .modules.EdgeProfiles import EdgeProfiles
# from .modules.EdgeSources import EdgeSources
# from .modules.EdgeTransport import EdgeTransport
# from .modules.EdgeTransportSolver import EdgeTransportSolver
# ---------------------------------

try:
    from .modules.Magnetics import Magnetics
    from .modules.PFActive import PFActive
    from .modules.TF import TF
    from .modules.Wall import Wall
except ModuleNotFoundError as error:
    raise RuntimeError(f"Fail to laod 'device' modules  ") from error

try:
    from .modules.ECLaunchers import ECLaunchers
    from .modules.ICAntennas import ICAntennas
    from .modules.LHAntennas import LHAntennas
    from .modules.NBI import NBI
    from .modules.Pellets import Pellets
except ModuleNotFoundError as error:
    raise RuntimeError(f"Fail to laod 'aux' modules  ") from error

try:
    from .modules.Interferometer import Interferometer
except ModuleNotFoundError as error:
    raise RuntimeError(f"Fail to laod 'diag' modules  ") from error

from .modules.TransportSolverNumerics import TransportSolverNumerics


class Tokamak(SpTree):
    """ Tokamak
        功能：
            - 描述装置在单一时刻的状态，
            - 在时间推进时，确定各个子系统之间的依赖和演化关系，

    """

    def __init__(self,  *args, device=None, shot=None, **kwargs):

        imas_version_major, *_ = imas_version.split(".")  # imas_version_minor, imas_version_patch,

        cache, entry, default_value, parent, kwargs = HTree._parser_args(*args, **kwargs)

        metadata = kwargs.pop("metadata", {})

        cache = merge_tree_recursive(cache, kwargs)

        cache["imas_version"] = imas_version
        cache["device"] = device
        cache["shot"] = shot or ""

        # if device is not None:
        #     entry = [f"{device}+://"]+entry

        entry = open_entry(entry, shot=shot, local_schema=device,  global_schema=f"imas/{imas_version_major}")

        super().__init__(cache, entry=entry,  parent=parent, default_value=default_value, metadata=metadata)

        self._device = device

    device: str = sp_property(default_value="unknown")

    shot: int = sp_property(default_value=0)

    time: float = sp_property(default_value=0.0)

    name: str = sp_property(default_value="unknown")

    description: str = sp_property(default_value="empty tokamak")

    # fmt:off

    # device
    wall                    : Wall                      = sp_property()

    # magnetics
    tf                      : TF                        = sp_property()
    pf_active               : PFActive                  = sp_property()
    magnetics               : Magnetics                 = sp_property()

    # aux
    ec_launchers            : ECLaunchers               = sp_property()
    ic_antennas             : ICAntennas                = sp_property()    
    lh_antennas             : LHAntennas                = sp_property()
    nbi                     : NBI                       = sp_property()
    pellets                 : Pellets                   = sp_property()

    # diag
    interferometer          : Interferometer            = sp_property()

    # transport: state of device
    equilibrium             : Equilibrium               = sp_property()

    core_profiles           : CoreProfiles              = sp_property()
    core_transport          : CoreTransport             = sp_property()
    core_sources            : CoreSources               = sp_property()

    # edge_profiles         : EdgeProfiles              = sp_property()
    # edge_transport        : EdgeTransport             = sp_property()
    # edge_sources          : EdgeSources               = sp_property()
    # edge_transport_solver : EdgeTransportSolver       = sp_property()

    # solver
    transport_solver        : TransportSolverNumerics   = sp_property()

    # fmt:on

    def advance(self, *args, **kwargs):

        self.equilibrium.advance(*args, **kwargs)

        self["time"] = self.equilibrium.time_slice.current.time

        # self.transport_solver.advance(*args, **kwargs,
        #                               equilibrium=self.equilibrium,
        #                               core_profiles=self.core_profiles,
        #                               core_sources=self.core_sources,
        #                               core_transport=self.core_transport,

        #                               )

    def refresh(self, *args, time=None, **kwargs):
        if time is not None:
            # 同步时间片
            self["time"] = time

            self.equilibrium.refresh(time=time)
            # self.core_profiles.refresh(time=time)
            # self.core_transport.refresh(time=time)
            # self.core_sources.refresh(time=time)

        # self.equilibrium.refresh(*args, **kwargs)

        # self.core_sources.refresh(*args, **kwargs)

        # self.core_transport.refresh(*args, **kwargs)

        # self.transport_solver.refresh(core_profiles=self.core_profiles,
        #                               equilibrium=self.equilibrium,
        #                               core_sources=self.core_sources,
        #                               core_transport=self.core_transport
        #                               )

    def __geometry__(self, view="RZ", **kwargs) -> GeoObject:

        # fmt:off
        geo = {
            "wall"          : self.wall.__geometry__(view=view, **kwargs),
            "tf"            : self.tf.__geometry__(view=view, **kwargs),
            "pf_active"     : self.pf_active.__geometry__(view=view, **kwargs),
            "magnetics"     : self.magnetics.__geometry__(view=view, **kwargs),


            ##################
            "ec_launchers"  : self.ec_launchers.__geometry__(view=view, **kwargs),
            "ic_antennas"   : self.ic_antennas.__geometry__(view=view, **kwargs),
            "lh_antennas"   : self.lh_antennas.__geometry__(view=view, **kwargs),
            "nbi"           : self.nbi.__geometry__(view=view, **kwargs),
            "pellets"       : self.pellets.__geometry__(view=view, **kwargs),
            "interferometer": self.interferometer.__geometry__(view=view, **kwargs),


            ##################
            "equilibrium"   : self.equilibrium.__geometry__(view=view, **kwargs),
        }
        # fmt:on

        if view != "RZ":
            styles = {
                "xlabel": r" $R$ [m]",
                "ylabel": r" $R$ [m]",
            }
        else:
            styles = {
                "xlabel": r"Major radius $R$ [m]",
                "ylabel": r"Height $Z$ [m]",
            }
        return geo, styles

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
