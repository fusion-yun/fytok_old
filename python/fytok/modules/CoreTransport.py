from __future__ import annotations

from copy import copy
import math
from spdm.data.AoS import AoS
from spdm.data.sp_property import sp_property, sp_tree
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.data.Expression import Expression
from spdm.utils.tags import _not_found_
from .Utilities import *
from .CoreProfiles import CoreProfiles
from .Equilibrium import Equilibrium
from ..utils.atoms import atoms
from ..utils.logger import logger

from ..ontology import core_transport


@sp_tree
class CoreTransportModelParticles(core_transport._T_core_transport_model_2_density):
    d: Expression = sp_property(coordinate1=".../grid_d/rho_tor_norm")
    v: Expression = sp_property(coordinate1=".../grid_v/rho_tor_norm")
    flux: Expression = sp_property(coordinate1=".../grid_flux/rho_tor_norm")


@sp_tree
class CoreTransportModelEnergy(core_transport._T_core_transport_model_2_energy):
    d: Expression = sp_property(coordinate1=".../grid_d/rho_tor_norm")
    v: Expression = sp_property(coordinate1=".../grid_v/rho_tor_norm")
    flux: Expression = sp_property(coordinate1=".../grid_flux/rho_tor_norm")


@sp_tree
class CoreTransportModelMomentum(core_transport._T_core_transport_model_4_momentum):
    d: Expression = sp_property(coordinate1=".../grid_d/rho_tor_norm")
    v: Expression = sp_property(coordinate1=".../grid_v/rho_tor_norm")
    flux: Expression = sp_property(coordinate1=".../grid_flux/rho_tor_norm")


@sp_tree
class CoreTransportElectrons(core_transport._T_core_transport_model_electrons):
    particles: CoreTransportModelParticles
    energy: CoreTransportModelEnergy
    momentum: CoreTransportModelMomentum


@sp_tree
class CoreTransportIon(core_transport._T_core_transport_model_ions):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.z is _not_found_:
            ion = atoms[self.label.capitalize()]
            self.z = ion.z
            self.a = ion.a

    label: str
    """ String identifying the neutral species (e.g. H, D, T, He, C, ...)"""

    z: int
    """ Charge number of the neutral species"""

    a: float
    """ Mass number of the neutral species"""

    particles: CoreTransportModelParticles
    energy: CoreTransportModelEnergy
    momentum: CoreTransportModelMomentum


@sp_tree
class CoreTransportNeutral(core_transport._T_core_transport_model_neutral):
    particles: CoreTransportModelParticles
    energy: CoreTransportModelEnergy


@sp_tree
class CoreTransportProfiles1D(core_transport._T_core_transport_model_profiles_1d):
    grid_d: CoreRadialGrid

    @sp_property
    def grid_v(self) -> CoreRadialGrid:
        return self.grid_d.remesh(self.grid_d.rho_tor_norm)

    @sp_property
    def grid_flux(self) -> CoreRadialGrid:
        rho_tor_norm = self.grid_d.rho_tor_norm
        return self.grid_d.remesh(0.5 * (rho_tor_norm[:-1] + rho_tor_norm[1:]))

    Electrons = CoreTransportElectrons

    Ion = CoreTransportIon

    Neutral = CoreTransportNeutral

    electrons: CoreTransportElectrons

    ion: AoS[CoreTransportIon] = sp_property(identifier="label")

    neutral: AoS[CoreTransportNeutral] = sp_property(identifier="label")


@sp_tree
class CoreTransportTimeSlice(TimeSlice):
    Profiles1D = CoreTransportProfiles1D

    vacuum_toroidal_field: VacuumToroidalField

    flux_multiplier: float = sp_property(default_value=0)

    profiles_1d: CoreTransportProfiles1D


@sp_tree
class CoreTransportModel(Module):
    _plugin_prefix = "fytok.plugins.core_transport.model."

    TimeSlice = CoreTransportTimeSlice

    identifier: str

    time_slice: TimeSeriesAoS[CoreTransportTimeSlice]

    def preprocess(self, *args, **kwargs):
        super().preprocess(*args, **kwargs)
        current = self.time_slice.current

        grid = current.cache_get("profiles_1d/grid_d", _not_found_)

        if not isinstance(grid, CoreRadialGrid):
            equilibrium: Equilibrium.TimeSlice = self.inputs.get_source("equilibrium").time_slice.current

            if current.time is _not_found_ or current.time is None:
                current.time = equilibrium.time

            assert math.isclose(equilibrium.time, self.time), f"{equilibrium.time} != {self.time}"

            current["profiles_1d/grid_d"] = equilibrium.profiles_1d.grid.remesh(
                grid,
                rho_tor_norm=self.code.parameters.get("rho_tor_norm", None),
            )

    def refresh(self, *args, core_profiles: CoreProfiles = None, equilibrium: Equilibrium = None, **kwargs):
        super().refresh(*args, core_profiles=core_profiles, equilibrium=equilibrium, **kwargs)


@sp_tree
class CoreTransport(IDS):
    Model = CoreTransportModel

    model: AoS[CoreTransportModel]

    def refresh(self, *args, equilibrium: Equilibrium = None, core_profiles: CoreProfiles = None, **kwargs):
        super().refresh(*args, **kwargs)

        for model in self.model:
            model.refresh(time=self.time, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

    def advance(self, *args, equilibrium: Equilibrium = None, core_profiles: CoreProfiles = None, **kwargs):
        """advance time_series to next slice"""
        super().advance(*args, **kwargs)

        for model in self.model:
            model.advance(time=self.time, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)
