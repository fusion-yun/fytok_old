import pathlib

import numpy as np
from fytok.Tokamak import Tokamak
from fytok.utils.plot_profiles import sp_figure
from spdm.data.File import File
from spdm.utils.logger import logger

###################


if __name__ == "__main__":

    logger.info("====== START ========")
    output_path = pathlib.Path('/home/salmon/workspace/output')

    ###################################################################################################
    # baseline
    device_desc = File("/home/salmon/workspace/fytok_data/mapping/ITER/imas/3/static/config.xml", format="XML").read()
    # Equilibrium
    eqdsk_file = File(
        "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt", format="GEQdsk").read()

    ###################################################################################################
    # Initialize Tokamak

    tok = Tokamak(device_desc[{"wall", "pf_active", "tf", "magnetics"}])
    tok["equilibrium"] = {**eqdsk_file.dump(),
                          "code": {"name":  "eq_analyze",
                                   "parameters": {
                                       "boundary": {"psi_norm": 0.995},
                                       "coordinate_system": {"psi_norm": np.linspace(0.001, 0.995, 64), "theta": 64}}
                                   }
                          }
    if True:
        sp_figure(tok,
                  wall={"limiter": {"edgecolor": "green"},
                        "vessel": {"edgecolor": "blue"}},
                  pf_active={"facecolor": 'red'},
                  #   equilibrium={
                  #       #   "contours": [0, 2],
                  #       "boundary": True,
                  #       "separatrix": True,
                  #   }
                  ) .savefig(output_path/"tokamak.svg", transparent=True)
