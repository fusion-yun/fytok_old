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
            ion = atoms[self.label]
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
    electrons: CoreTransportElectrons

    Ion = CoreTransportIon
    ion: AoS[CoreTransportIon] = sp_property(identifier="label", default_initial={})

    Neutral = CoreTransportNeutral
    neutral: AoS[CoreTransportNeutral] = sp_property(identifier="label", default_initial={})


@sp_tree
class CoreTransportTimeSlice(TimeSlice):
    vacuum_toroidal_field: VacuumToroidalField

    flux_multiplier: float = sp_property(default_value=0)

    Profiles1D = CoreTransportProfiles1D
    profiles_1d: CoreTransportProfiles1D


@sp_tree
class CoreTransportModel(Module):
    _plugin_prefix = "fytok.plugins.core_transport.model."

    identifier: str

    TimeSlice = CoreTransportTimeSlice
    time_slice: TimeSeriesAoS[CoreTransportTimeSlice]

    def preprocess(self, *args, **kwargs) -> CoreTransportTimeSlice:
        current = super().preprocess(*args, **kwargs)

        grid = current.find_cache("profiles_1d/grid_d", _not_found_)

        if not isinstance(grid, CoreRadialGrid):
            equilibrium: Equilibrium.TimeSlice = self.inputs.get_source("equilibrium").time_slice.current

            rho_tor_norm = kwargs.get("rho_tor_norm", self.code.parameters.get("rho_tor_norm", None))

            current["profiles_1d/grid_d"] = equilibrium.profiles_1d.grid.remesh(grid, rho_tor_norm=rho_tor_norm)

    def fetch(self, *args, **kwargs) -> CoreTransportTimeSlice:
        current: CoreTransportTimeSlice = super().fetch(*args, **kwargs)

        # grid = current.profiles_1d.find_cache("grid_d", _not_found_)

        # if grid is _not_found_:
        #     current.profiles_1d["grid_d"] = profiles_1d.grid

        # else:
        #     grid = current.profiles_1d.grid_d
        #     if grid.psi_axis is _not_found_ or grid.psi_axis is None:
        #         grid["psi_axis"] = profiles_1d.grid.psi_axis
        #         grid["psi_boundary"] = profiles_1d.grid.psi_boundary
        #         grid["rho_tor_boundary"] = profiles_1d.grid.rho_tor_boundary

        # x = profiles_1d.rho_tor_norm

        # current = current.fetch(x)

        # if isinstance(x, array_type):
        #     current.profiles_1d["grid_d"] = profiles_1d.grid

        return current

    def flush(self) -> CoreTransportTimeSlice:
        super().flush()

        current = self.time_slice.current

        profiles_1d: CoreProfiles.TimeSlice.Profiles1D = self.inports["core_profiles/time_slice/0/profiles_1d"].fetch()

        current.update(self.fetch(profiles_1d))

        return current

    def refresh(
        self,
        *args,
        core_profiles: CoreProfiles = None,
        equilibrium: Equilibrium = None,
        **kwargs,
    ) -> CoreTransportTimeSlice:
        return super().refresh(*args, core_profiles=core_profiles, equilibrium=equilibrium, **kwargs)


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

    def flush(self):
        super().flush()
        for model in self.model:
            model.flush()
