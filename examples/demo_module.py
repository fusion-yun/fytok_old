import pathlib

import numpy as np
import pandas as pd
from fytok.transport.Equilibrium import Equilibrium
from spdm.data import File, Function, Query
from spdm.util.logger import logger

if __name__ == "__main__":
    logger.info("====== START ========")
    output_path = pathlib.Path('/home/salmon/workspace/output')
    eqdsk_file = File(
        "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt", format="geqdsk")

    eq_desc = {**eqdsk_file.entry.dump(),
               "code": {"name": "dummy"},
               "boundary": {"psi_norm": 0.995},
               "coordinate_system": {"psi_norm": np.linspace(0.001, 0.995, 64), "theta": 64}}

    eq = Equilibrium(eq_desc)

    logger.debug(eq)
