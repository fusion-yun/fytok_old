from copy import copy

from spdm.data.AoS import AoS
from spdm.data.sp_property import sp_property, sp_tree
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.data.Expression import Expression
from spdm.utils.tags import _not_found_
from .Utilities import *
from .CoreProfiles import CoreProfiles
from .Equilibrium import Equilibrium
from ..ontology import core_transport
from ..utils.logger import logger


@sp_tree
class CoreTransportModelParticles(core_transport._T_core_transport_model_2_density):
    d: Expression = 0
    v: Expression = 0
    flux: Expression = 0


@sp_tree
class CoreTransportModelEnergy(core_transport._T_core_transport_model_2_energy):
    d: Expression = 0
    v: Expression = 0
    flux: Expression = 0


@sp_tree
class CoreTransportModelMomentum(core_transport._T_core_transport_model_4_momentum):
    d: Expression = 0
    v: Expression = 0
    flux: Expression = 0


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


@sp_tree(coordinate1="grid_d/rho_tor_norm")
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

    TimeSlice = CoreTransportTimeSlice

    identifier: str

    time_slice: TimeSeriesAoS[CoreTransportTimeSlice]

    def refresh(self, *args, **kwargs):
        super().refresh(*args, **kwargs)
        if self.time_slice.current.profiles_1d.get("grid", _not_found_) is _not_found_ and "equilibrium" in kwargs:
            equilibrium: Equilibrium = kwargs["equilibrium"]
            grid = copy(equilibrium.time_slice.current.profiles_1d.grid).remesh(kwargs.pop("rho_tor_norm", None))
            self.time_slice.current.profiles_1d["grid_d"] = grid


@sp_tree
class CoreTransport(core_transport._T_core_transport):
    Model = CoreTransportModel

    model: AoS[CoreTransportModel]

    def refresh(self, *args, **kwargs):
        for model in self.model:
            model.refresh(*args, **kwargs)

    def advance(self, *args, **kwargs):
        """advance time_series to next slice"""
        for model in self.model:
            model.advance(*args, **kwargs)
