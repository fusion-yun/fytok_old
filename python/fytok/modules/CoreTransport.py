from spdm.data.AoS import AoS
from spdm.data.sp_property import sp_property, sp_tree
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.data.Expression import Expression

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

    def refresh(self, *args, core_profiles: CoreProfiles.TimeSlice, equilibrium: Equilibrium.TimeSlice, **kwargs):
        """update the last time slice"""

        super().refresh(
            {
                "time": core_profiles.time,
                "vacuum_toroidal_field": core_profiles.vacuum_toroidal_field,
                "profiles_1d": {
                    "grid_d": core_profiles.profiles_1d.grid,
                    "ion": [{"label": ion.label} for ion in core_profiles.profiles_1d.ion],
                    "neutral": [{"label": neutral.label} for neutral in core_profiles.profiles_1d.neutral],
                },
            },
            *args,
            **kwargs,
        )

    def fetch(self, /, x, **vars) -> CoreTransportTimeSlice:
        res = CoreTransportTimeSlice({"profiles_1d": {}})

        res_1d = res.profiles_1d

        core_trans_1d = self.time_slice.current.profiles_1d

        res_1d.electrons["particles"] = (
            {
                "d": core_trans_1d.electrons.particles.d(x),
                "v": core_trans_1d.electrons.particles.v(x),
                "flux": core_trans_1d.electrons.particles.flux(x),
            },
        )
        res_1d.electrons["energy"] = {
            "d": core_trans_1d.electrons.energy.d(x),
            "v": core_trans_1d.electrons.energy.v(x),
            "flux": core_trans_1d.electrons.energy.flux(x),
        }

        res_1d["ion"] = [
            {
                "label": ion.label,
                "particles": {
                    "d": ion.particles.d(x),
                    "v": ion.particles.v(x),
                    "flux": ion.particles.flux(x),
                },
                "energy": {
                    "d": ion.energy.d(x),
                    "v": ion.energy.v(x),
                    "flux": ion.energy.flux(x),
                },
            }
            for ion in core_trans_1d.ion
        ]

        return res


@sp_tree
class CoreTransport(core_transport._T_core_transport):
    Model = CoreTransportModel

    model: AoS[CoreTransportModel]

    def refresh(self, *args, **kwargs):
        """update the last time slice"""

        for model in self.model:
            model.refresh(*args, **kwargs)

    def advance(self, *args, **kwargs):
        """advance time_series to next slice"""
        for model in self.model:
            model.advance(*args, **kwargs)
