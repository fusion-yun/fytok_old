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
        return self.grid_d.duplicate(self.grid_d.rho_tor_norm)

    @sp_property
    def grid_flux(self) -> CoreRadialGrid:
        rho_tor_norm = self.grid_d.rho_tor_norm
        return self.grid_d.duplicate(0.5 * (rho_tor_norm[:-1] + rho_tor_norm[1:]))

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

    code: Code = {"name": "dummy"}

    TimeSlice = CoreTransportTimeSlice

    identifier: str

    time_slice: TimeSeriesAoS[CoreTransportTimeSlice]

    def preprocess(self, *args, **kwargs):
        super().preprocess(*args, **kwargs)

        current = self.time_slice.current

        if current.cache_get("grid_d", _not_found_) is _not_found_:
            equilibrium: Equilibrium.TimeSlice = self.inputs.get_source("equilibrium").time_slice.current

            if current.time is _not_found_ or current.time is None:
                current.time = equilibrium.time

            assert math.isclose(equilibrium.time, self.time), f"{equilibrium.time} != {self.time}"

            current["profiles_1d/grid_d"] = equilibrium.profiles_1d.grid.remesh(
                self.code.parameters.get("rho_tor_norm", None)
            )

    def refresh(self, *args, core_profiles: CoreProfiles = None, equilibrium: Equilibrium = None, **kwargs):
        super().refresh(*args, core_profiles=core_profiles, equilibrium=equilibrium, **kwargs)

        # current = self.time_slice.current.profiles_1d
        # if current.cache_get("grid", _not_found_) is _not_found_:
        #     equilibrium: Equilibrium = self._inputs.get("equilibrium")
        #     rho_tor_norm = self.code.parameters.get("rho_tor_norm", None)
        #     current["grid"] = equilibrium.time_slice.current.profiles_1d.grid.duplicate(rho_tor_norm)


@sp_tree
class CoreTransport(core_transport._T_core_transport):
    Model = CoreTransportModel

    model: AoS[CoreTransportModel]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def refresh(
        self,
        *args,
        equilibrium: Equilibrium = None,
        core_profiles: CoreProfiles = None,
        **kwargs,
    ):
        super().refresh(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

        for model in self.model:
            model.refresh(time=self.time, **self._inputs)

    def advance(
        self,
        *args,
        equilibrium: Equilibrium = None,
        core_profiles: CoreProfiles = None,
        **kwargs,
    ):
        """advance time_series to next slice"""
        super().advance(*args, equilibrium=equilibrium, core_profiles=core_profiles, **kwargs)

        for model in self.model:
            model.advance(time=self.time, **self._inputs)
