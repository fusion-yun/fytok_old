
from fytok._imas.lastest.core_sources import (_T_core_sources,
                                              _T_core_sources_source)
from spdm.data.List import List
from spdm.data.sp_property import SpDict, sp_property

from .CoreProfiles import CoreProfiles
from .Utilities import CoreRadialGrid


class CoreSourcesSource(_T_core_sources_source):
    _IDS = "core_sources/source"

    @property
    def grid(self):
        return self._parent.grid

    def update(self, *args, **kwargs) -> float:
        residual = super().update(*args,  **kwargs)
        return residual


class CoreSources(_T_core_sources):

    Source = CoreSourcesSource

    grid: CoreRadialGrid = sp_property()

    source: List[Source] = sp_property()

    @property
    def source_combiner(self) -> Source:
        return self.source.combine(
            common_data={
                "identifier": {"name": "total", "index": 1,
                               "description": "Total source; combines all sources"},
                "code": {"name": None},
            })

    def update(self,  core_profiles: CoreProfiles, *args, **kwargs) -> float:
        self["grid"] = core_profiles.profiles_1d.grid
        for src in self.source:
            src.refresh(*args, core_profiles=core_profiles, **kwargs)
