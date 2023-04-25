import pathlib

import numpy as np
from spdm.data.File import File
from spdm.utils.logger import logger
from spdm.view.plot_profiles import sp_figure

from fytok.modules.Equilibrium import Equilibrium

###################


if __name__ == "__main__":
    logger.info("====== START ========")
    output_path = pathlib.Path('/home/salmon/workspace/output')

    # Equilibrium
    eqdsk_file = File(
        "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt", format="GEQdsk").read()

    eq = Equilibrium(eqdsk_file, code={"name": "fy_eq",
                                       "parameters": {
                                           "boundary": {"psi_norm": 0.995},
                                           "coordinate_system": {"psi_norm": np.linspace(0.001, 0.995, 64), "theta": 64}}
                                       })

    sp_figure(eq,
              contours=20,
              boundary=True,
              separatrix=True,
              ) .savefig(output_path/"eq.svg", transparent=True)
