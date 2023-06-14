import os
import pathlib

import numpy as np
from fytok.Tokamak import Tokamak
from fytok.utils.load_scenario import load_scenario
from fytok.utils.plot_profiles import plot_profiles, sp_figure
from scipy import constants
from spdm.data.Expression import Variable
from spdm.data.File import File
from spdm.utils.logger import logger

if __name__ == "__main__":

    logger.info("====== START ========")

    output_path = pathlib.Path('/home/salmon/workspace/output')

    scenario = load_scenario("/home/salmon/workspace/data/15MA inductive - burn")

    tok = Tokamak("ITER",
                  name=scenario["name"],
                  description=scenario["description"],
                  core_profiles={**scenario["core_profiles"],
                                 #  "$default_value": {
                                 #      "profiles_1d": {"grid": {
                                 #          "rho_tor_norm": np.linspace(0, 1.0, 100),
                                 #          "psi": np.linspace(0, 1.0, 100),
                                 #          "psi_magnetic_axis": 0.0,
                                 #          "psi_boundary": 1.0,
                                 #      }}}
                                 },
                  equilibrium={**scenario["equilibrium"],
                               "code": {"name":  "freegs",
                                        "parameters": {
                                            "boundary": "fixed",
                                            "psi_norm": np.linspace(0, 1.0, 128)
                                        }
                                        },
                               "$default_value": {"time_slice": {
                                   "boundary": {"psi_norm": 0.99},
                                   "profiles_2d": {"grid": {"dim1": 256, "dim2": 128}},
                                   "coordinate_system": {"grid": {"dim1": 256, "dim2": 128}}
                               }}}
                  )
    if False:
        sp_figure(tok,
                  wall={"limiter": {"edgecolor": "green"},
                        "vessel": {"edgecolor": "blue"}},
                  pf_active={"color": 'red'},
                  equilibrium={  # "contours": [0, 2],
                      "boundary": True,
                      "separatrix": True,
                  }
                  ) .savefig(output_path/"tokamak_prev.svg", transparent=True)

    psi_bndry = None # tok.equilibrium.time_slice.current.boundary.psi

    logger.debug(psi_bndry)

    tok.equilibrium.update(
        wall=tok.wall,
        pf_active=tok.pf_active,
        Ip=1.5e6, beta_p=0.6056,
        psi_bndry=psi_bndry,
        tolerance=1.0e-2,)

    if True:
        sp_figure(tok,
                  wall={"limiter": {"edgecolor": "green"},
                        "vessel": {"edgecolor": "blue"}},
                  pf_active={"color": 'red'},
                  equilibrium={  # "contours": [0, 2],
                      "boundary": False,
                      "separatrix": False,
                  }
                  ) .savefig(output_path/"tokamak_post.svg", transparent=True)
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
