
from spdm.data.AoS import AoS
from spdm.data.sp_property import sp_property, sp_tree
from spdm.data.TimeSeries import TimeSeriesAoS


from ..utils.logger import logger
from .Utilities import *
from .CoreProfiles import CoreProfiles
from .Equilibrium import Equilibrium

from .._ontology import core_transport


@sp_tree
class CoreTransportElectrons(core_transport._T_core_transport_model_electrons):
    pass


@sp_tree
class CoreTransportIon(core_transport._T_core_transport_model_ions):
    pass


@sp_tree
class CoreTransportNeutral(core_transport._T_core_transport_model_neutral):
    pass


@sp_tree
class CoreTransportProfiles1D(core_transport._T_core_transport_model_profiles_1d):

    Electrons = CoreTransportElectrons

    Ion = CoreTransportIon

    Neutral = CoreTransportNeutral

    grid_d: CoreRadialGrid

    @sp_property
    def grid_v(self) -> CoreRadialGrid:
        return self.grid_d.remesh(self.grid_d.rho_tor_norm)

    @sp_property
    def grid_flux(self) -> CoreRadialGrid:
        rho_tor_norm = self.grid_d.rho_tor_norm
        return self.grid_d.remesh(0.5*(rho_tor_norm[:-1]+rho_tor_norm[1:]))

    electrons: Electrons

    ion: AoS[Ion]

    neutral: AoS[Neutral]


@sp_tree
class CoreTransportTimeSlice:

    Profiles1D = CoreTransportProfiles1D

    vacuum_toroidal_field: VacuumToroidalField

    profiles_1d: Profiles1D


@sp_tree
class CoreTransport(IDS):

    @sp_tree
    class Model(Module):
        _plugin_prefix = 'fytok.plugins.core_transport.model.'

        identifier: str

        TimeSlice = CoreTransportTimeSlice

        time_slice: TimeSeriesAoS[CoreTransportTimeSlice]

    model: AoS[Model]
