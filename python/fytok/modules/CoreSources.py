
from _imas import _T_core_sources
from _imas.core_sources import _T_core_sources_source
from spdm.data.List import List
from spdm.data.sp_property import sp_property

from .Module import Module
from .CoreProfiles import CoreProfiles
from .MagneticCoordSystem import RadialGrid


class CoreSourcesSource(_T_core_sources_source, Module):
    _IDS = "core_sources/source"

    @property
    def grid(self):
        return self._parent.grid

    def refresh(self, *args, **kwargs) -> float:
        residual = super().refresh(*args,  **kwargs)
        return residual


class CoreSources(_T_core_sources):

    grid: RadialGrid = sp_property()

    source: List[CoreSourcesSource] = sp_property()

    @property
    def source_combiner(self) -> CoreSourcesSource:
        return self.source.combine(
            common_data={
                "identifier": {"name": "total", "index": 1,
                               "description": "Total source; combines all sources"},
                "code": {"name": None},
            })

    def refresh(self,  core_profiles: CoreProfiles, *args, **kwargs) -> float:
        self["grid"] = core_profiles.profiles_1d.grid
        for src in self.source:
            src.refresh(*args, core_profiles=core_profiles, **kwargs)
