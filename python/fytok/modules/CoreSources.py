from __future__ import annotations
from copy import copy
import math
from spdm.data.AoS import AoS
from spdm.data.sp_property import sp_property, sp_tree
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.data.Expression import Expression, zero, one
from spdm.utils.tags import _not_found_

from .CoreProfiles import CoreProfiles
from .Equilibrium import Equilibrium
from .Utilities import *
from ..utils.atoms import atoms

from ..ontology import core_sources


@sp_tree
class CoreSourcesSpecies(SpTree):
    """Source terms related to electrons"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        atom_desc = atoms.get(self.label.capitalize(), None)

        self._cache = update_tree(self._cache, atom_desc)

    label: str
    """ String identifying the neutral species (e.g. H, D, T, He, C, ...)"""

    z: int
    """ Charge number of the neutral species"""

    a: float
    """ Mass number of the neutral species"""

    particles: Expression= sp_property(units="s^-1.m^-3")
    """Source term for electron density equation"""
    

    @sp_property(units="s^-1")
    def particles_inside(self) -> Expression:
        """Electron source inside the flux surface. Cumulative volume integral of the
        source term for the electron density equation."""
        return self.particles.I

    @sp_property(units="W.m^-3")
    def energy(self) -> Expression:
        """Source term for the electron energy equation"""
        value = self.cache_get("energy", _not_found_)
        if value is _not_found_:
            if self.label == "e":
                label = "e"
            else:
                label = f"ion/{self.label}"

            temperature = self.get(f".../inputs/core_profiles/{label}/temperature", zero)

            value = (
                self.cache_get("particles_decomposed/explicit_part", 0)
                + self.cache_get("particles_decomposed/implicit_part", 0) * temperature
            )

        return value

    @sp_property(units="W")
    def power_inside(self) -> Expression:
        """Power coupled to electrons inside the flux surface. Cumulative volume integral
        of the source term for the electron energy equation"""
        return self.energy.I


@sp_tree
class CoreSourcesNeutral(core_sources._T_core_sources_source_profiles_1d_neutral):
    pass


@sp_tree(coordinate1="grid/rho_tor_norm")
class CoreSourcesProfiles1D(core_sources._T_core_sources_source_profiles_1d):
    grid: CoreRadialGrid
    """ Radial grid"""

    total_ion_energy: Expression = sp_property(units="W.m^-3")
    """Total ion energy source"""

    @sp_property(units="W")
    def total_ion_power_inside(self) -> Expression:
        return self.torque_tor_inside.I

    momentum_tor: Expression

    torque_tor_inside: Expression

    momentum_tor_j_cross_b_field: Expression

    j_parallel: Expression

    current_parallel_inside: Expression

    conductivity_parallel: Expression

    electrons: CoreSourcesSpecies

    ion: AoS[CoreSourcesSpecies]

    neutral: AoS[CoreSourcesNeutral]


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

    def preprocess(self, *args, **kwargs):
        super().preprocess(*args, **kwargs)

        current = self.time_slice.current

        if current.cache_get("grid", _not_found_) is _not_found_:
            equilibrium: Equilibrium.TimeSlice = self.inputs.get_source("equilibrium").time_slice.current

            if current.time is _not_found_ or current.time is None:
                current.time = equilibrium.time

            assert math.isclose(equilibrium.time, self.time), f"{equilibrium.time} != {self.time}"

            current["profiles_1d/grid"] = equilibrium.profiles_1d.grid.remesh(
                self.code.parameters.get("rho_tor_norm", None)
            )

    def refresh(self, *args, equilibrium: Equilibrium = None, core_profiles: CoreProfiles = None, **kwargs):
        super().refresh(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

    def fetch(self, /, x: Expression, **variables) -> CoreSourcesTimeSlice:
        res: CoreSourcesTimeSlice = super().fetch(lambda o: o if not isinstance(o, Expression) else o(x))

        res["electrons/particles"] = res.cache_get("particles", _not_found_)
        
        if value is _not_found_:
            if self.label == "e":
                label = "e"
            else:
                label = f"ion/{self.label}"

            density = self.get(f".../core_profiles/{label}/density", zero)

            value = (
                self.cache_get("particles_decomposed/explicit_part", 0)
                + self.cache_get("particles_decomposed/implicit_part", 0) * density
            )

        return value
        return res


@sp_tree
class CoreSources(IDS):
    Source = CoreSourcesSource

    source: AoS[CoreSourcesSource]

    def refresh(self, *args, equilibrium: Equilibrium = None, core_profiles: CoreProfiles = None, **kwargs):
        super().refresh(*args, **kwargs)

        for source in self.source:
            source.refresh(time=self.time, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

    def advance(self, *args, equilibrium: Equilibrium = None, core_profiles: CoreProfiles = None, **kwargs):
        super().advance(*args, **kwargs)

        for source in self.source:
            source.advance(time=self.time, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)
