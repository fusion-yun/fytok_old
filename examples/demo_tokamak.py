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
        core_profiles_1d = tok.core_profiles.profiles_1d.current

        display(
            [
                core_profiles_1d.ffprime,
                core_profiles_1d.pprime,
                ([
                    (core_profiles_1d.electrons.density, {"label": "n_e"}),
                    *[(ion.density, {"label": ion.label}) for ion in core_profiles_1d.ion if not ion.is_impurity],
                ], {"y_label": "Density [$m^{-3}$]"}),
            ],
            x_axis=(core_profiles_1d.grid.rho_tor_norm, {"x_label": r"$\rho=\sqrt{\Phi/\Phi_{bdry}}$"}),
            title=" CoreProfile initialize value",
            output=output_path/"core_profiles_initialize.svg",
        )
    if False:

        boundary_outline_r = tok.equilibrium.time_slice.current.boundary.outline.r
        boundary_outline_z = tok.equilibrium.time_slice.current.boundary.outline.z
        boundary_psi = np.full_like(boundary_outline_r, tok.equilibrium.time_slice.current.boundary.psi)
        psivals = np.vstack([boundary_outline_r, boundary_outline_z, boundary_psi]).T

        xpoints = [(x.r, x.z) for x in tok.equilibrium.time_slice.current.boundary.x_point]

        tok.equilibrium.update(
            # profiles
            core_profiles_1d=tok.core_profiles.profiles_1d.current,
            Ip=1.5e6, beta_p=0.6056,

            # constrain
            # xpoints=xpoints,
            psivals=psivals,

            # options
            tolerance=1.0e-1,)

        display(tok,
                # styles={"equilibrium": {"boundary": False, "boundary_separatrix": False, }},
                xlabel=r"$R$ [m]",
                ylabel=r"$Z$ [m]",
                output=output_path/"tokamak_post.svg",
                transparent=True)

    logger.info("DONE")
