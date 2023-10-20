
from spdm.data.AoS import AoS
from spdm.data.sp_property import sp_property, sp_tree
from spdm.data.TimeSeries import TimeSeriesAoS


from .Utilities import *
from .CoreProfiles import CoreProfiles
from .Equilibrium import Equilibrium
from ..utils.logger import logger
from ..ontology import core_transport


@sp_tree
class CoreTransportModelParticles(core_transport._T_core_transport_model_2_density):
    d: Function = sp_property(coordinate1="../../../grid_d/rho_tor_norm", units="m^2.s^-1", default_value=np.nan)
    v: Function = sp_property(coordinate1="../../../grid_v/rho_tor_norm", units="m.s^-1", default_value=np.nan)
    flux: Function = sp_property(coordinate1="../../../grid_flux/rho_tor_norm",
                                 units="m^-2.s^-1", default_value=np.nan)


@sp_tree
class CoreTransportModelEnergy(core_transport._T_core_transport_model_2_energy):
    d: Function = sp_property(coordinate1="../../../grid_d/rho_tor_norm", units="m^2.s^-1", default_value=np.nan)
    v: Function = sp_property(coordinate1="../../../grid_v/rho_tor_norm", units="m.s^-1", default_value=np.nan)
    flux: Function = sp_property(coordinate1="../../../grid_flux/rho_tor_norm", units="W.m^-2", default_value=np.nan)


@sp_tree
class CoreTransportModelMomentum(core_transport._T_core_transport_model_4_momentum):
    d: Function = sp_property(coordinate1="../../../grid_d/rho_tor_norm", units="m^2.s^-1", default_value=np.nan)
    v: Function = sp_property(coordinate1="../../../grid_v/rho_tor_norm", units="m.s^-1", default_value=np.nan)
    flux: Function = sp_property(coordinate1="../../../grid_flux/rho_tor_norm", units="W.m^-2", default_value=np.nan)


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

    Electrons = CoreTransportElectrons

    Ion = CoreTransportIon

    Neutral = CoreTransportNeutral

    grid_d: CoreRadialGrid

    @sp_property
    def grid_v(self) -> CoreRadialGrid: return self.grid_d.remesh(self.grid_d.rho_tor_norm)

    @sp_property
    def grid_flux(self) -> CoreRadialGrid:
        rho_tor_norm = self.grid_d.rho_tor_norm
        return self.grid_d.remesh(0.5*(rho_tor_norm[:-1]+rho_tor_norm[1:]))

    electrons: Electrons

    ion: AoS[Ion]

    neutral: AoS[Neutral]


@sp_tree
class CoreTransportTimeSlice(TimeSlice):

    Profiles1D = CoreTransportProfiles1D

    vacuum_toroidal_field: VacuumToroidalField

    profiles_1d: CoreTransportProfiles1D


@sp_tree
class CoreTransportModel(TimeBasedActor):

    _plugin_prefix = 'fytok.plugins.core_transport.model.'

    TimeSlice = CoreTransportTimeSlice

    identifier: str

    time_slice: TimeSeriesAoS[CoreTransportTimeSlice]

    def refresh(self, *args,
                core_profiles: CoreProfiles.TimeSlice,
                equilibrium: Equilibrium.TimeSlice,
                ** kwargs):
        """update the last time slice"""

        super().refresh(
            {
                "time": core_profiles.time,
                "vacuum_toroidal_field": core_profiles.vacuum_toroidal_field,
                "profiles_1d": {
                    "grid_d": core_profiles.profiles_1d.grid,
                    "ion": [{"label": ion.label} for ion in core_profiles.profiles_1d.ion],
                    "neutral": [{"label": neutral.label} for neutral in core_profiles.profiles_1d.neutral]
                }
            },
            *args, ** kwargs)


@ sp_tree
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
