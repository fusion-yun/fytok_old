
from __future__ import annotations


from spdm.data.AoS import AoS
from spdm.data.sp_property import sp_property
from spdm.data.TimeSeries import TimeSeriesAoS
from ..utils.logger import logger
from spdm.utils.typing import array_type
from ..schema import core_sources
from .CoreProfiles import CoreProfiles
from .Equilibrium import Equilibrium
from ..utils.utilities import CoreRadialGrid


class CoreSourcesSource(core_sources._T_core_sources_source):
    _plugin_prefix = 'fytok.plugins.core_sources.source.'
    _plugin_config = {}

    Profiles1D = core_sources._T_core_sources_source_profiles_1d

    GlobalQuantities = core_sources._T_core_sources_source_global

    @property
    def time(self) -> array_type: return self._parent.time

    profiles_1d: TimeSeriesAoS[Profiles1D] = sp_property()

    global_quantities: TimeSeriesAoS[GlobalQuantities] = sp_property()

    def refresh(self, *args, **kwargs):
        logger.debug(f"{self.__class__.__name__}.refresh")

    def advance(self, *args, **kwargs):
        logger.debug(f"{self.__class__.__name__}.advance")


class CoreSources(core_sources._T_core_sources):

    Source = CoreSourcesSource

    grid: CoreRadialGrid = sp_property()

    source: AoS[Source] = sp_property(default_value={
        "identifier": {"name": "total", "index": 1,
                               "description": "Total source; combines all sources"},
        "code": {"name": None},
    })

    def advance(self, *args, equilibrium: Equilibrium.TimeSlice, core_profiles_1d: CoreProfiles.Profiles1D, **kwargs):
        for source in self.source:
            source.advance(*args, equilibrium=equilibrium, core_profiles_1d=core_profiles_1d, **kwargs)

    def refresh(self, *args, equilibrium: Equilibrium.TimeSlice, core_profiles_1d: CoreProfiles.Profiles1D, **kwargs):
        for source in self.source:
            source.refresh(*args, equilibrium=equilibrium, core_profiles_1d=core_profiles_1d, **kwargs)
