from __future__ import annotations

from spdm.data.AoS import AoS
from spdm.data.sp_property import sp_property, sp_tree
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.data.Expression import Expression


from .Utilities import *
from ..ontology import core_sources


@sp_tree
class CoreSourcesElectrons(core_sources._T_core_sources_source_profiles_1d_electrons):
    pass


@sp_tree
class CoreSourcesIon(core_sources._T_core_sources_source_profiles_1d_ions):
    pass


@sp_tree
class CoreSourcesNeutral(core_sources._T_core_sources_source_profiles_1d_neutral):
    pass


@sp_tree(coordinate1="grid/rho_tor_norm", default_value=0)
class CoreSourcesProfiles1D(core_sources._T_core_sources_source_profiles_1d):
    grid: CoreRadialGrid

    """ Radial grid"""

    electrons: CoreSourcesElectrons
    """ Sources for electrons"""

    total_ion_energy: Expression = sp_property(units="W.m^-3")
    """ Source term for the total (summed over ion species) energy equation"""

    total_ion_energy_decomposed: core_sources._T_core_sources_source_profiles_1d_energy_decomposed_2

    total_ion_power_inside: Expression = sp_property(units="W")

    momentum_tor: Expression = sp_property(units="kg.m^-1.s^-2")

    torque_tor_inside: Expression = sp_property(units="kg.m^2.s^-2")

    momentum_tor_j_cross_b_field: Expression = sp_property(units="kg.m^-1.s^-2")

    j_parallel: Expression = sp_property(units="A.m^-2")

    current_parallel_inside: Expression = sp_property(units="A")

    conductivity_parallel: Expression = sp_property(units="ohm^-1.m^-1")

    ion: AoS[CoreSourcesIon] = sp_property(identifier="label")

    neutral: AoS[CoreSourcesNeutral] = sp_property(identifier="label")


@sp_tree
class CoreSourcesGlobalQuantities(core_sources._T_core_sources_source_global):
    pass


@sp_tree
class CoreSourcesTimeSlice(TimeSlice):
    Profiles1D = CoreSourcesProfiles1D

    GlobalQuantities = CoreSourcesGlobalQuantities

    profiles_1d: CoreSourcesProfiles1D

    global_quantities: CoreSourcesGlobalQuantities


@sp_tree
class CoreSourcesSource(Module):
    _plugin_prefix = "fytok.plugins.core_sources.source."

    identifier: str

    species: DistributionSpecies

    TimeSlice = CoreSourcesTimeSlice

    time_slice: TimeSeriesAoS[CoreSourcesTimeSlice]

    def fetch(self, *args, **kwargs) -> CoreSourcesTimeSlice:
        return super().fetch(*args, **kwargs)


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
