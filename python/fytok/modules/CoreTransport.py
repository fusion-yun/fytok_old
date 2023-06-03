from functools import cached_property

from fytok._imas.lastest.core_transport import (
    _T_core_transport, _T_core_transport_model, _T_core_transport_model_ions, _T_core_transport_model_neutral,
    _T_core_transport_model_profiles_1d)
from spdm.data.List import List, AoS
from spdm.data.sp_property import SpDict, sp_property
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.data.Entry import deep_reduce

from .CoreProfiles import CoreProfiles
from .Utilities import CoreRadialGrid


class CoreTransportProfiles1D(_T_core_transport_model_profiles_1d):

    grid_d: CoreRadialGrid = sp_property()

    @sp_property
    def grid_v(self) -> CoreRadialGrid:
        return self.grid_d.remesh("rho_tor_norm", self.grid_d.rho_tor_norm)

    @sp_property
    def grid_flux(self) -> CoreRadialGrid:
        return self.grid_d.remesh("rho_tor_norm", 0.5*(self.grid_d.rho_tor_norm[:-1]+self.grid_d.rho_tor_norm[1:]))


class CoreTransportModel(_T_core_transport_model):

    _IDS = "core_transport/model"
    CoreProfiles1D = CoreTransportProfiles1D

    profiles_1d: TimeSeriesAoS[CoreProfiles1D] = sp_property()

    def advance(self, *args, core_profiles: CoreProfiles, time: float = 0.0, **kwargs) -> CoreProfiles1D:
        return self.profiles_1d.advance(*args, core_profiles=core_profiles, time=time, **kwargs)


class CoreTransport(_T_core_transport):

    Model = CoreTransportModel

    model: AoS[Model] = sp_property()

    @property
    def model_combiner(self) -> CoreTransportModel:
        return self.model.combine(
            default_value={
                "identifier": {"name": "combined", "index": 1,
                               "description": """Combination of data from available transport models.
                                Representation of the total transport in the system"""},
                "code": {"name": None},
            })

    def advance(self, *args, **kwargs) -> Model.CoreProfiles1D:
        return CoreTransportModel.CoreProfiles1D(deep_reduce([model.advance(*args, **kwargs) for model in self.model]), parent=self)
