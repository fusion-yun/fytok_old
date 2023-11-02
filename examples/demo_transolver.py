import os
import numpy as np

from fytok.modules.Equilibrium import Equilibrium
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreTransport import CoreTransport
from fytok.modules.CoreSources import CoreSources
from fytok.modules.TransportSolverNumerics import TransportSolverNumerics
from fytok.utils.logger import logger
from fytok.utils.load_scenario import load_scenario
from spdm.view import View as sp_view

if __name__ == "__main__":
    WORKSPACE = "/home/salmon/workspace"

    # f"{WORKSPACE}/gacode/neo/tools/input/profile_data"
    input_path = "/home/salmon/workspace/fytok_data/data/15MA inductive - burn"

    output_path = f"{WORKSPACE}/output"

    scenario = load_scenario(input_path)

    equilibrium = Equilibrium(scenario["equilibrium"])

    core_profiles = CoreProfiles(scenario["core_profiles"])

    core_transport = CoreTransport(scenario["core_transport"])

    core_sources = CoreSources(scenario["core_sources"])

    trans = TransportSolverNumerics({"code":  {"name": "fytrans", }})

    trans.refresh(
        equilibrium=equilibrium,
        core_profiles=core_profiles,
        core_transport=core_transport,
        core_sources=core_sources,
    )

    # eq_profiles_1d = equilibrium.time_slice.current.profiles_1d

    # core_profiles_1d: CoreProfiles.TimeSlice.Profiles1D = core_profiles.time_slice.current.profiles_1d

    # logger.debug(core_profiles_1d.electrons.density(core_profiles_1d.grid.rho_tor_norm))

    # for ion in core_profiles_1d.ion:
    #     logger.debug(ion.label)
    #     logger.debug(ion.density(core_profiles_1d.grid.rho_tor_norm))
    #     # logger.debug(ion.temperature(core_profiles_1d.grid.rho_tor_norm))

    # sp_view.plot([
    #     ([
    #         (core_profiles_1d.electrons.density, {"label": r"$n_e$"}),
    #         *[(ion.density, {"label": rf"$n_{ion.label}$"}) for ion in core_profiles_1d.ion]
    #     ], {"y_label": r"$[m^{-3}]$"}),
    #     ([
    #         (core_profiles_1d.electrons.temperature, {"label": r"$n_e$"}),
    #         *[(ion.temperature, {"label": rf"$T_{ion.label}$"}) for ion in core_profiles_1d.ion]
    #     ], {"y_label": r"$[eV]$"}),

    # ],
    #     x_value=core_profiles_1d.grid.rho_tor_norm,
    #     x_label=r"$\bar{\rho}_{tor}$",
    #     stop_if_fail=False,
    #     output=f"{output_path}/core_profiles.svg")
    # logger.debug(tgyro.model[0].current.profiles_1d.grid_d.rho_tor_norm)
