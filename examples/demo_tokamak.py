import os
import pathlib

import numpy as np
import scipy.constants
from fytok.Tokamak import Tokamak
from fytok.utils.load_scenario import load_scenario
from spdm.data.Expression import Variable
from spdm.data.File import File
from spdm.utils.logger import logger
from spdm.views.View import display

if __name__ == "__main__":

    logger.info("====== START ========")

    output_path = pathlib.Path('/home/salmon/workspace/output')

    scenario = load_scenario("/home/salmon/workspace/data/15MA inductive - burn")

    tok = Tokamak("ITER",
                  name=scenario["name"],
                  description=scenario["description"],
                  core_profiles={
                      **scenario["core_profiles"],
                      "$default_value": {
                          "profiles_1d": {
                              "grid": {
                                  "rho_tor_norm": np.linspace(0, 1.0, 100),
                                  "psi": np.linspace(0, 1.0, 100),
                                  "psi_magnetic_axis": 0.0,
                                  "psi_boundary": 1.0,
                              }}}
                  },
                  equilibrium={
                      **scenario["equilibrium"],
                      "code": {
                          "name":  "freegs",
                          "parameters": {
                              "boundary": "fixed",
                              "psi_norm": np.linspace(0, 1.0, 128)
                          }},
                      "$default_value": {
                          "time_slice": {
                              "boundary": {"psi_norm": 0.99},
                              "profiles_2d": {"grid": {"dim1": 256, "dim2": 128}},
                              "coordinate_system": {"grid": {"dim1": 256, "dim2": 128}}
                          }}}
                  )

    # logger.debug(tok.equilibrium.time_slice.current.profiles_1d.j_tor._repr_latex_())

    if True:
        display(tok, title=f"{tok.name} time={tok.time}s", output=output_path/"tokamak_prev.svg")

    if True:
        # psirz = tok.equilibrium.time_slice.current.profiles_2d[0].psi.__array__()
        # psi2 = np.linspace(np.min(psirz), np.max(psirz), 100)
        # pprime = tok.equilibrium.time_slice.current.profiles_1d.pprime(psi2)
        # logger.debug(pprime)

        boundary_outline_r = tok.equilibrium.time_slice.current.boundary.outline.r
        boundary_outline_z = tok.equilibrium.time_slice.current.boundary.outline.z
        boundary_psi = np.full_like(boundary_outline_r, tok.equilibrium.time_slice.current.boundary.psi)
        psivals = np.vstack([boundary_outline_r, boundary_outline_z, boundary_psi]).T

        xpoints = [(x.r, x.z) for x in tok.equilibrium.time_slice.current.boundary.x_point]

        tok.equilibrium.update(
            # machine
            wall=tok.wall, pf_active=tok.pf_active, magnetic=tok.magnetics,

            # profiles
            core_profiles_1d=tok.core_profiles.profiles_1d.current,
            Ip=1.5e6, beta_p=0.6056,

            # constrain
            xpoints=xpoints,
            psivals=psivals,

            # options
            tolerance=1.0e-1,)

        display(tok,
                styles={"wall": {"limiter": {"edgecolor": "green"}, "vessel": {"edgecolor": "blue"}},
                        "pf_active": {"color": 'red'},
                        "equilibrium": {"boundary": True, "separatrix": True, }},
                output=output_path/"tokamak_post.svg",
                transparent=True)

    if False:
        core_profile_1d = tok.core_profiles.profiles_1d.current
        plot_profiles(
            [
                (core_profile_1d .ffprime, "$ff'$", "$ff'$"),
                (core_profile_1d .pprime, "$p'$", "$p'$"),
            ],
            x_axis=(core_profile_1d.grid.rho_tor_norm, r"$\rho=\sqrt{\Phi/\Phi_{bdry}}$"),
            grid=True, fontsize=10) .savefig(output_path/"core_profiles_initialize.svg", transparent=True)

    logger.info("DONE")
