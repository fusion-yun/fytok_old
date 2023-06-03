from functools import cached_property

from fytok._imas.lastest.core_transport import (
    _T_core_transport, _T_core_transport_model, _T_core_transport_model_ions, _T_core_transport_model_neutral,
    _T_core_transport_model_profiles_1d)
from spdm.data.List import List, AoS
from spdm.data.sp_property import SpDict, sp_property
from spdm.data.TimeSeries import TimeSeriesAoS

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

    profiles_1d: TimeSeriesAoS[CoreTransportProfiles1D] = sp_property()

    def update(self, *args, core_profiles: CoreProfiles, **kwargs) -> None:
        super().update(*args, core_profiles=core_profiles, **kwargs)
        # self.profiles_1d["grid"] = core_profiles.profiles_1d.grid


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

    def update(self, *args, core_profiles=None, **kwargs) -> None:
        if "model_combiner" in self.__dict__:
            del self.__dict__["model_combiner"]

        if core_profiles is not None:
            self["grid"] = core_profiles.profiles_1d.grid

        for model in self.model:
            model.refresh(*args, core_profiles=core_profiles, **kwargs)
