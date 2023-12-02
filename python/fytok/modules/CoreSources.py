from __future__ import annotations
from copy import copy
from spdm.data.AoS import AoS
from spdm.data.sp_property import sp_property, sp_tree
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.data.Expression import Expression
from spdm.utils.tags import _not_found_

from .CoreProfiles import CoreProfiles
from .Equilibrium import Equilibrium
from .Utilities import *

from ..ontology import core_sources


@sp_tree
class CoreSourcesElectrons(core_sources._T_core_sources_source_profiles_1d_electrons):
    particles_decomposed = {"implicit_part": 0, "explicit_part": 0}
    energy_decomposed = {"implicit_part": 0, "explicit_part": 0}
    particles: Expression = sp_property(label="S_{e}", default_value=0)
    energy: Expression = sp_property(label="S_{e}", default_value=0)


@sp_tree
class CoreSourcesIon(core_sources._T_core_sources_source_profiles_1d_ions):
    particles_decomposed = {"implicit_part": 0, "explicit_part": 0}
    energy_decomposed = {"implicit_part": 0, "explicit_part": 0}
    particles: Expression = sp_property(label="S_{i}", default_value=0)


@sp_tree
class CoreSourcesNeutral(core_sources._T_core_sources_source_profiles_1d_neutral):
    pass


@sp_tree(coordinate1="grid/rho_tor_norm")
class CoreSourcesProfiles1D(core_sources._T_core_sources_source_profiles_1d):
    grid: CoreRadialGrid
    """ Radial grid"""

    electrons: CoreSourcesElectrons = {}

    total_ion_energy: Expression

    total_ion_energy_decomposed = {"implicit_part": 0, "explicit_part": 0}

    total_ion_power_inside: Expression

    momentum_tor: Expression

    torque_tor_inside: Expression

    momentum_tor_j_cross_b_field: Expression

    j_parallel: Expression

    current_parallel_inside: Expression

    conductivity_parallel: Expression

    ion: AoS[CoreSourcesIon] = sp_property(identifier="label", default_value=[])

    neutral: AoS[CoreSourcesNeutral] = sp_property(identifier="label", default_value=[])


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

    def __init__(self, *args, _parent: CoreSources = None, **kwargs):
        if isinstance(_parent, AoS):
            _parent = _parent._parent
        super().__init__(*args, _parent=_parent, **kwargs)

    identifier: str

    species: DistributionSpecies

    TimeSlice = CoreSourcesTimeSlice

    time_slice: TimeSeriesAoS[CoreSourcesTimeSlice]

    def refresh(self, *args, equilibrium: Equilibrium = None, core_profiles: CoreProfiles = None, **kwargs):
        super().refresh(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

        # current = self.time_slice.current.profiles_1d
        # if current.cache_get("grid_d", _not_found_) is _not_found_:
        #     equilibrium: Equilibrium = self._inputs.get("equilibrium")
        #     rho_tor_norm = self.code.parameters.get("rho_tor_norm", None)
        #     current["grid"] = equilibrium.time_slice.current.profiles_1d.grid.duplicate(rho_tor_norm)

    def fetch(self, /, x: Expression, **vars) -> CoreSourcesTimeSlice:
        res: CoreSourcesTimeSlice = super().fetch(lambda o: o if not isinstance(o, Expression) else o(x))

        res_1d = res.profiles_1d

        res_1d.electrons["particles"] = (
            res_1d.electrons.particles
            + res_1d.electrons.particles_decomposed.implicit_part * vars.get("electrons/density_thermal", 0)
            + res_1d.electrons.particles_decomposed.explicit_part
        )

        res_1d.electrons["energy"] = (
            res_1d.electrons.energy
            + res_1d.electrons.energy_decomposed.implicit_part * vars.get("electrons/temperature", 0)
            + res_1d.electrons.energy_decomposed.explicit_part
        )

        for ion in res_1d.ion:
            ion["particles"] = (
                ion.particles
                + ion.particles_decomposed.implicit_part * vars.get(f"ion/{ion.label}/density_thermal", 0)
                + ion.particles_decomposed.explicit_part
            )

            ion["energy"] = (
                ion.energy
                + ion.energy_decomposed.implicit_part * vars.get(f"ion/{ion.label}/temperature", 0)
                + ion.energy_decomposed.explicit_part
            )

        return res


@sp_tree
class CoreSources(IDS):
    Source = CoreSourcesSource

    source: AoS[CoreSourcesSource]

    def refresh(self, *args, equilibrium: Equilibrium = None, core_profiles: CoreProfiles = None, **kwargs):
        super().refresh(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

        for source in self.source:
            source.refresh(time=self.time, **self._inputs)

    def advance(self, *args, equilibrium: Equilibrium = None, core_profiles: CoreProfiles = None, **kwargs):
        super().advance(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

        for source in self.source:
            source.advance(time=self.time, **self._inputs)
