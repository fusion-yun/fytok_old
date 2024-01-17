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
    label: str = "electrons"
    """ String identifying the neutral species (e.g. H, D, T, He, C, ...)"""

    particles: CoreTransportModelParticles
    energy: CoreTransportModelEnergy
    momentum: CoreTransportModelMomentum


@sp_tree
class CoreTransportIon(core_transport._T_core_transport_model_ions):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        ion = atoms[self.label]
        self.z = ion.z
        self.a = ion.a

    label: str = sp_property(alias="@name")
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

        grid = current.get_cache("profiles_1d/grid_d", _not_found_)

        if not isinstance(grid, CoreRadialGrid):
            eq_grid: CoreRadialGrid = self.inports["equilibrium/time_slice/0/profiles_1d/grid"].fetch()

            if isinstance(grid, dict):
                new_grid = grid
                if not isinstance(grid.get("psi_axis", _not_found_), float):
                    new_grid["psi_axis"] = eq_grid.psi_axis
                    new_grid["psi_boundary"] = eq_grid.psi_boundary
                    new_grid["rho_tor_boundary"] = eq_grid.rho_tor_boundary
                # new_grid = {
                #     **eq_grid._cache,
                #     **{k: v for k, v in grid.items() if v is not _not_found_ and v is not None},
                # }
            else:
                rho_tor_norm = kwargs.get("rho_tor_norm", self.code.parameters.rho_tor_norm)
                new_grid = eq_grid.remesh(rho_tor_norm)

            current["profiles_1d/grid_d"] = new_grid

        return current

    def fetch(self, *args, **kwargs) -> CoreTransportTimeSlice:
        if len(args) > 0 and isinstance(args[0], CoreProfiles.TimeSlice.Profiles1D):
            args = (args[0].rho_tor_norm, *args[1:])

        return super().fetch(*args, **kwargs)

    def flush(self) -> CoreTransportTimeSlice:
        super().flush()

        current = self.time_slice.current

        profiles_1d: CoreProfiles.TimeSlice.Profiles1D = self.inports["core_profiles/time_slice/0/profiles_1d"].fetch()

        current.update(self.fetch(profiles_1d)._cache)

        return current

    def refresh(
        self, *args, core_profiles: CoreProfiles = None, equilibrium: Equilibrium = None, **kwargs
    ) -> CoreTransportTimeSlice:
        return super().refresh(*args, core_profiles=core_profiles, equilibrium=equilibrium, **kwargs)


@sp_tree
class CoreTransport(IDS):
    Model = CoreTransportModel

    model: AoS[CoreTransportModel]

    def initialize(self, *args, **kwargs) -> None:
        super().initialize(*args, **kwargs)
        for model in self.model:
            model.initialize()

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
