
from __future__ import annotations

import typing

from spdm.data.Entry import deep_reduce
from spdm.data.List import AoS, List
from spdm.data.sp_property import SpDict, sp_property
from spdm.data.TimeSeries import TimeSeriesAoS

from .._imas.lastest.core_sources import (_T_core_sources,
                                          _T_core_sources_source,
                                          _T_core_sources_source_global,
                                          _T_core_sources_source_profiles_1d)
from .CoreProfiles import CoreProfiles
from .Utilities import CoreRadialGrid


class CoreSourcesSource(_T_core_sources_source):
    _IDS = "core_sources/source"

    Profiles1d = _T_core_sources_source_profiles_1d

    GlobalQuantities = _T_core_sources_source_global

    profiles_1d: TimeSeriesAoS[Profiles1d] = sp_property()
    
    global_quantities: TimeSeriesAoS[GlobalQuantities] = sp_property()

    def update(self, *args, **kwargs) -> Profiles1d:
        profiles_1d = self.profiles_1d.update(*args,  **kwargs)
        self.global_quantities.update()
        return profiles_1d

    def advance(self, *args, **kwargs) -> Profiles1d:
        profiles_1d = self.profiles_1d.advance(*args, **kwargs)
        self.global_quantities.advance()
        return profiles_1d


class CoreSources(_T_core_sources):

    Source = CoreSourcesSource

    grid: CoreRadialGrid = sp_property()

    source: AoS[Source] = sp_property()

    @property
    def source_combiner(self) -> Source:
        return self.source.combine(
            default_value={
                "identifier": {"name": "total", "index": 1,
                               "description": "Total source; combines all sources"},
                "code": {"name": None},
            })

    def advance(self, *args, **kwargs) -> Source.Profiles1D:
        return CoreSources.Source.Profiles1D(deep_reduce([source.advance(*args, **kwargs) for source in self.source]), parent=self)

    def update(self, *args, **kwargs) -> Source.Profiles1D:
        return CoreSources.Source.Profiles1D(deep_reduce([source.update(*args, **kwargs) for source in self.source]), parent=self)
