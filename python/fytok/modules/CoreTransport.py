import typing
from functools import cached_property

from fytok._imas.lastest.core_transport import (
    _T_core_transport, _T_core_transport_model, _T_core_transport_model_ions,
    _T_core_transport_model_neutral, _T_core_transport_model_profiles_1d)
from spdm.data.Entry import as_entry, deep_reduce
from spdm.data.Function import Function
from spdm.data.HTree import AoS, List
from spdm.data.sp_property import SpDict, sp_property
from spdm.data.TimeSeries import TimeSeriesAoS
from spdm.utils.logger import logger
from spdm.utils.tags import _not_found_
from spdm.utils.tree_utils import merge_tree_recursive

from .CoreProfiles import CoreProfiles
from .Equilibrium import Equilibrium
from .Utilities import CoreRadialGrid


class CoreTransportNeutral(_T_core_transport_model_neutral):
    pass


class CoreTransportIon(_T_core_transport_model_ions):
    """Transport coefficients for density equations. Coordinates two levels above."""
    pass


class CoreTransportProfiles1D(_T_core_transport_model_profiles_1d):

    Ion = CoreTransportIon

    Neutral = CoreTransportNeutral

    grid_d: CoreRadialGrid = sp_property()

    @sp_property
    def grid_v(self) -> CoreRadialGrid:
        return self.grid_d.remesh("rho_tor_norm", self.grid_d.rho_tor_norm)

    @sp_property
    def grid_flux(self) -> CoreRadialGrid:
        return self.grid_d.remesh("rho_tor_norm", 0.5*(self.grid_d.rho_tor_norm[:-1]+self.grid_d.rho_tor_norm[1:]))

    ion: AoS[Ion] = sp_property(coordinate1="1...N", identifier="label")

    neutral: AoS[Neutral] = sp_property(coordinate1="1...N")


class CoreTransportModel(_T_core_transport_model):
    _plugin_registry = {}

    _IDS = "core_transport/model"

    Profiles1D = CoreTransportProfiles1D

    @property
    def time(self): return self._parent.time

    profiles_1d: TimeSeriesAoS[Profiles1D] = sp_property()

    def refresh(self, *args, **kwargs):
        logger.debug(f"{self.__class__.__name__}.refresh")

    def advance(self, *args, **kwargs):
        logger.debug(f"{self.__class__.__name__}.advance")


class CoreTransport(_T_core_transport):

    Model = CoreTransportModel

    model: AoS[Model] = sp_property()

    def advance(self, *args, equilibrium: Equilibrium.TimeSlice, core_profiles_1d: CoreProfiles.Profiles1d, **kwargs):
        for model in self.model:
            model.advance(*args, equilibrium=equilibrium, core_profiles_1d=core_profiles_1d, **kwargs)

    def refresh(self, *args,  equilibrium: Equilibrium.TimeSlice, core_profiles_1d: CoreProfiles.Profiles1d, **kwargs):
        for model in self.model:
            model.refresh(*args, equilibrium=equilibrium, core_profiles_1d=core_profiles_1d, **kwargs)
