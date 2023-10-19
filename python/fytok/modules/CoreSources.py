
from __future__ import annotations

from spdm.data.AoS import AoS
from spdm.data.sp_property import sp_property, sp_tree
from spdm.data.TimeSeries import TimeSeriesAoS

from ..utils.logger import logger

from .CoreProfiles import CoreProfiles
from .Equilibrium import Equilibrium
from .Utilities import *
from ..ontology import core_sources


@sp_tree
class CoreSourceTimeSlice(TimeSlice):

    @sp_tree
    class Profiles1D(core_sources._T_core_sources_source_profiles_1d):
        grid: CoreRadialGrid

    @sp_tree
    class GlobalQuantities(core_sources._T_core_sources_source_global):
        pass

    profiles_1d: Profiles1D

    global_quantities: GlobalQuantities


@sp_tree
class CoreSourcesSource(TimeBasedActor):

    _plugin_prefix = 'fytok.plugins.core_sources.source.'

    identifier: str

    species: DistributionSpecies

    TimeSlice = CoreSourceTimeSlice

    time_slice: TimeSeriesAoS[CoreSourceTimeSlice]


@sp_tree
class CoreSources(IDS):

    Source = CoreSourcesSource

    source: AoS[CoreSourcesSource]

    def refresh(self, *args, **kwargs):
        for source in self.source:
            source.refresh(*args, **kwargs)

    def advance(self, *args, **kwargs):
        for source in self.source:
            source.advance(*args, **kwargs)
