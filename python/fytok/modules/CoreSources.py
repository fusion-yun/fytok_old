
from __future__ import annotations

from spdm.data.AoS import AoS
from spdm.data.sp_property import sp_property, sp_tree
from spdm.data.TimeSeries import TimeSeriesAoS

from ..utils.logger import logger

from .CoreProfiles import CoreProfiles
from .Equilibrium import Equilibrium
from .Utilities import *


@sp_tree
class CoreSourceTimeSlice(TimeSlice):

    @sp_tree(coordinate1="grid/rho_tor_norm", bind="core_sources.source.prorfiles_1d")
    class Profiles1D(TimeSlice):
        grid: CoreRadialGrid

    @sp_tree(bind="core_sources.source.global_quantities")
    class GlobalQuantities(TimeSlice):
        pass

    profiles_1d: Profiles1D

    global_quantities: GlobalQuantities


@sp_tree
class CoreSources(IDS):

    @sp_tree
    class Source(Module):

        _plugin_prefix = 'fytok.plugins.core_sources.source.'

        identifier: str

        species: DistributionSpecies

        TimeSlice = CoreSourceTimeSlice

        time_slice: TimeSeriesAoS[CoreSourceTimeSlice]

    source: AoS[Source]
