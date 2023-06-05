import pathlib
import os
import numpy as np
import pandas as pd
from fytok.Tokamak import Tokamak
from fytok.utils.plot_profiles import plot_profiles, sp_figure
from scipy import constants
from spdm.data.Expression import Variable
from spdm.data.File import File
from spdm.data.Function import function_like
from spdm.utils.logger import logger

os.environ["SP_DATA_MAPPING_PATH"] = "/home/salmon/workspace/fytok_data/mapping"

if __name__ == "__main__":

    logger.info("====== START ========")

    output_path = pathlib.Path('/home/salmon/workspace/output')

    # f_path = "file+MDSplus[EAST]:///home/salmon/workspace/data/~t/?tree_name=efit_east#38300"
    eqdsk_file = File(
        "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt", format="GEQdsk").read()

    tok = Tokamak("ITER",
                  name="ITER 15MA 9T",
                  description="ITER 15MA 9T",
                  equilibrium={
                    #   **eqdsk_file.dump(),
                    #   "time": [0.0],
                      "vacuum_toroidal_field": {"r0": 6.2, "b0": [-5.3]},
                      "code": {"name":  "eq_analyze", },
                      "$default_value": {
                          "time_slice": {
                              "profiles_2d": {"grid": {"dim1": 129, "dim2": 257}},
                              "boundary": {"psi_norm": 0.99},
                              "coordinate_system": {"grid": {"dim1": 256, "dim2": 128}}
                          }}}
                  )

    # tok.equilibrium.advance(time=0.0)

    sp_figure(tok,
              wall={"limiter": {"edgecolor": "green"},
                    "vessel": {"edgecolor": "blue"}},
              pf_active={"color": 'red'},
              equilibrium={  # "contours": [0, 2],
                  "boundary": True,
                  "separatrix": True,
              }
              ) .savefig(output_path/"tokamak.svg", transparent=True)

    logger.info("DONE")
