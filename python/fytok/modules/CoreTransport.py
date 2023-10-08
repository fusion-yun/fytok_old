
from spdm.data.AoS import AoS
from spdm.data.sp_property import sp_property
from spdm.data.TimeSeries import TimeSeriesAoS

from .schema import core_transport

from ..utils.logger import logger
from ..utils.utilities import CoreRadialGrid
from .CoreProfiles import CoreProfiles
from .Equilibrium import Equilibrium


class CoreTransportElectrons(core_transport._T_core_transport_model_electrons):

    momentum: core_transport._T_core_transport_model_1_momentum = sp_property()
    """ Transport coefficients related to the ion momentum equations for various
		components (directions)"""


class CoreTransportNeutral(core_transport._T_core_transport_model_neutral):
    pass


class CoreTransportIon(core_transport._T_core_transport_model_ions):
    """Transport coefficients for density equations. Coordinates two levels above."""
    pass


class CoreTransportProfiles1D(core_transport._T_core_transport_model_profiles_1d):

    Ion = CoreTransportIon

    Electrons = CoreTransportElectrons

    Neutral = CoreTransportNeutral

    grid_d: CoreRadialGrid = sp_property()

    @property
    def grid_v(self) -> CoreRadialGrid: return self.grid_d.remesh(self.grid_d.rho_tor_norm)

    @property
    def grid_flux(self) -> CoreRadialGrid:
        rho_tor_norm = self.grid_d.rho_tor_norm
        return self.grid_d.remesh(0.5*(rho_tor_norm[:-1]+rho_tor_norm[1:]))

    electrons: Electrons = sp_property()

    ion: AoS[Ion] = sp_property(coordinate1="1...N", identifier="label")

    neutral: AoS[Neutral] = sp_property(coordinate1="1...N")


class CoreTransportModel(core_transport._T_core_transport_model):
    _plugin_prefix = 'fytok.plugins.core_transport.model.'
    _plugin_config = {}

    Profiles1D = CoreTransportProfiles1D

    profiles_1d: TimeSeriesAoS[Profiles1D] = sp_property()

    def refresh(self, cache=None,   **kwargs):
        logger.debug(f"{self.__class__.__name__}.refresh")
        self.profiles_1d.refresh(cache.pop("profiles_1d", None) if cache is not None else None,   **kwargs)

    def advance(self, *args, **kwargs):
        logger.debug(f"{self.__class__.__name__}.advance")
        self.profiles_1d.advance(*args, **kwargs)


class CoreTransport(core_transport._T_core_transport):

    Model = CoreTransportModel

    model: AoS[Model] = sp_property()

    def advance(self, *args,   **kwargs):
        for model in self.model:
            model.advance(*args,   **kwargs)

    def refresh(self, *args,   **kwargs):
        for model in self.model:
            model.refresh(*args,   **kwargs)
