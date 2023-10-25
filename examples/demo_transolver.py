import os
import numpy as np
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.TransportSolverNumerics import TransportSolverNumerics
from fytok.modules.Equilibrium import Equilibrium
from fytok.utils.logger import logger
from spdm.view import View as sp_view

if __name__ == "__main__":
    WORKSPACE = "/home/salmon/workspace"

    input_path = f"{WORKSPACE}/gacode/neo/tools/input/profile_data"

    output_path = f"{WORKSPACE}/output"

    equilibrium = Equilibrium(f"file+geqdsk://{input_path}/g141459.03890#equilibrium")

    profiles_1d = equilibrium.time_slice.current.profiles_1d

    core_profiles = CoreProfiles(f"file+iterdb://{input_path}/iterdb141459.03890#core_profiles")

    trans = TransportSolverNumerics(
        {"code":  {"name": "fytrans",
                   "parameters": {"input_file": "/home/salmon/workspace/gacode/tgyro/tools/input/iter01/input.gacode", "sim_model": 1}, },

         "$default_value": {"model/time_slice/profiles_1d/grid_d/rho_tor_norm": np.linspace(0, 1, 100), }
         })

    trans.refresh(
        core_profiles=core_profiles.time_slice.current,
        equilibrium=equilibrium.time_slice.current,
    )

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
