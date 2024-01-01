from __future__ import annotations

from spdm.data.Path import update_tree
from spdm.data.Entry import open_entry
from spdm.data.HTree import HTree
from spdm.data.Actor import Actor
from spdm.data.sp_property import sp_tree
from spdm.data.TimeSeries import TimeSlice, TimeSeriesAoS
from spdm.geometry.GeoObject import GeoObject
from spdm.utils.tags import _not_found_
from spdm.view import View as sp_view

# ---------------------------------
from .modules.DatasetFAIR import DatasetFAIR
from .modules.Summary import Summary
from .modules.CoreProfiles import CoreProfiles
from .modules.CoreSources import CoreSources
from .modules.CoreTransport import CoreTransport
from .modules.ECLaunchers import ECLaunchers
from .modules.Equilibrium import Equilibrium
from .modules.ICAntennas import ICAntennas
from .modules.Interferometer import Interferometer
from .modules.LHAntennas import LHAntennas
from .modules.Magnetics import Magnetics
from .modules.NBI import NBI
from .modules.Pellets import Pellets
from .modules.PFActive import PFActive
from .modules.TF import TF
from .modules.Wall import Wall
from .modules.TransportSolverNumerics import TransportSolverNumerics

from .utils.envs import *
from .utils.logger import logger
from .ontology import GLOBAL_ONTOLOGY

# from .modules.EdgeProfiles import EdgeProfiles
# from .modules.EdgeSources import EdgeSources
# from .modules.EdgeTransport import EdgeTransport
# from .modules.EdgeTransportSolver import EdgeTransportSolver
# ---------------------------------


@sp_tree
class Tokamak(Actor):
    def __init__(
        self,
        *args,
        device: str = _not_found_,
        shot: int = _not_found_,
        run: int = _not_found_,
        time: float = None,
        **kwargs,
    ):
        """
        用于集成子模块，以实现工作流。

        现有子模块包括： wall, tf, pf_active, magnetics, equilibrium, core_profiles, core_transport, core_sources, transport_solver

        :param args:   初始化数据，可以为 dict，str 或者  Entry。 输入会通过数据集成合并为单一的HTree，其子节点会作为子模块的初始化数据。
        :param device: 指定装置名称，例如， east，ITER, d3d 等
        :param shot:   指定实验炮号
        :param run:    指定模拟计算的序号
        :param time:   指定当前时间
        :param kwargs: 指定子模块的初始化数据，，会与args中指定的数据源子节点合并。
        """
        cache, entry, parent, kwargs = HTree._parser_args(*args, **kwargs)

        cache = update_tree(cache, kwargs)

        cache["dataset_fair"] = {"description": {"entry": entry, "device": device, "shot": shot or 0, "run": run or 0}}

        entry = open_entry(entry, shot=shot, run=run, local_schema=device, global_schema=GLOBAL_ONTOLOGY)

        super().__init__(cache, _entry=entry, _parent=parent)

        self._shot = shot
        self._run = run
        self._device = device

    @property
    def brief_summary(self) -> str:
        """综述模拟内容"""
        return f"""{FY_LOGO}
---------------------------------------------------------------------------------------------------
                                                Brief Summary
---------------------------------------------------------------------------------------------------
Dataset Description:
{self.dataset_fair}
---------------------------------------------------------------------------------------------------
Modules:
    transport_solver        : {self.transport_solver.code }
    equilibrium             : {self.equilibrium.code }

    core_profiles           : N/A             
    core_transport          : {','.join([str(s.code) for s in self.core_transport.model])}
    core_sources            : {','.join([str(s.code)  for s in self.core_sources.source])}
---------------------------------------------------------------------------------------------------
"""

    # Data source:
    #     {pprint.pformat(str(self._entry).split(','))}
    # ---------------------------------------------------------------------------------------------------

    # File: {__file__}:{__package__}.{self.__class__.__name__}

    # edge_profiles           : N/A
    # edge_transport          : N/A
    # edge_sources            : N/A
    # edge_transport_solver   : N/A
    @property
    def title(self) -> str:
        """标题，由初始化信息 dataset_fair.description"""
        return f"{self.dataset_fair.description}  time={self.time:.2f}s"

    @property
    def tag(self) -> str:
        """当前状态标签，由程序版本、用户名、时间戳等信息确定"""
        return f"{self.dataset_fair.description.tag}_{int(self.time*100):06d}"

    @property
    def shot(self) -> int:
        return self._shot

    @property
    def run(self) -> int:
        return self._run

    @property
    def device(self) -> str:
        return self._device

    # fmt:off
    # device
    dataset_fair            : DatasetFAIR               

    wall                    : Wall                      

    # magnetics
    tf                      : TF                        
    pf_active               : PFActive                  
    magnetics               : Magnetics                 

    # aux
    ec_launchers            : ECLaunchers               
    ic_antennas             : ICAntennas                    
    lh_antennas             : LHAntennas                
    nbi                     : NBI                       
    pellets                 : Pellets                   

    # diag
    interferometer          : Interferometer            

    # transport: state of device
    equilibrium             : Equilibrium               

    core_profiles           : CoreProfiles              
    core_transport          : CoreTransport             
    core_sources            : CoreSources               

    # edge_profiles         : EdgeProfiles              
    # edge_transport        : EdgeTransport             
    # edge_sources          : EdgeSources               
    # edge_transport_solver : EdgeTransportSolver       

    # solver
    transport_solver        : TransportSolverNumerics   

    summary                 : Summary                   
    # fmt:on

    def setup(self, *args, **kwargs):
        fusion_products = set()
        fusion_ash = set()
        for tag in self.fusion_reactions:
            reaction = nuclear_reaction[tag]
            for r in reaction.reactants:
                if r not in ["electrons", "e", "n", "p", "alpha"]:
                    r = r.capitalize()
                self.ion.add(r)
            for p in reaction.products:
                if p not in ["electrons", "e", "n", "p", "alpha"]:
                    p = r.capitalize()
                if p == "n":
                    continue

                ash = atoms[p].label
                if ash == p:
                    raise NotImplementedError(f"TODO: give unified identifier to fast particle")
                fusion_products.add(p)
                fusion_ash.add(ash)

    def advance(self, *args, **kwargs):
        return super().advance(*args, **kwargs)

    def refresh(self, *args, **kwargs) -> None:
        super().refresh(*args, **kwargs)

        self.equilibrium.refresh(time=self.time, **self._inputs.fetch())

        self.core_sources.refresh(time=self.time, **self._inputs.fetch())

        self.core_transport.refresh(time=self.time, **self._inputs.fetch())

    def flush(self, *args, **kwargs):
        super().flush(*args, **kwargs)

        profiles_1d = self.transport_solver.fetch()

        self.core_profiles.flush(profiles_1d=profiles_1d)

        self.equilibrium.flush()

        self.core_transport.flush()

        self.core_sources.flush()

        self.transport_solver.flush()

    def __geometry__(self, **kwargs) -> GeoObject:
        geo = {}

        o_list = [
            "wall",
            "equilibrium",
            "pf_active",
            "magnetics",
            "interferometer",
            "tf",
            # "ec_launchers",
            # "ic_antennas",
            # "lh_antennas",
            # "nbi",
            # "pellets",
        ]

        for o_name in o_list:
            try:
                g = getattr(self, o_name, None)
                if g is None:
                    continue
                g = g.__geometry__(**kwargs)

            except Exception as error:
                logger.error(f"Can not get {o.__class__.__name__}.__geometry__ ! {error}")
                # raise RuntimeError(f"Can not get {g.__class__.__name__}.__geometry__ !") from error
            else:
                geo[o_name] = g

        view_point = (kwargs.get("view_point", None) or "rz").lower()

        styles = {}

        if view_point == "rz":
            styles["xlabel"] = r"Major radius $R [m] $"
            styles["ylabel"] = r"Height $Z [m]$"

        styles["title"] = kwargs.pop("title", None) or self.title

        geo["$styles"] = styles

        return geo

    def _repr_svg_(self):
        try:
            res = sp_view.display(self.__geometry__(), output="svg")
        except Exception as error:
            raise RuntimeError(f"{self}") from error
            # res = None
        return res

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
