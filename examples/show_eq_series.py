import os
import pathlib

import numpy as np
from spdm.core.Entry import Entry
from spdm.core.File import File
from fytok.utils.logger import logger
from spdm.view.View import display

from fytok.modules.Equilibrium import Equilibrium
from fytok.Tokamak import Tokamak

WORKSPACE = "/home/salmon/workspace"  # "/ssd01/salmon_work/workspace/"

os.environ["SP_DATA_MAPPING_PATH"] = f"{WORKSPACE}/fytok_data/mapping"


if __name__ == "__main__":
    output_path = pathlib.Path(f"{WORKSPACE}/output/")

    shot = 70745

    tok = Tokamak(C)

    tok.refresh(time=9.26)

    psi = tok.equilibrium.time_slice.current.profiles_2d[0].psi.__array__()

    psi_min = psi.min()

    psi_max = psi.max()

    levels = np.arange(psi_min, psi_max, (psi_max-psi_min)/40)

    for i in range(150):
        display(tok.equilibrium,
                title=f"EAST shot={shot} time={tok.equilibrium.time_slice.current.time:.2f}s ",
                output=output_path / f"{tok.tag}.png",
                transparent=False,
                styles={"psi": {"$matplotlib": {"levels": levels}}}
                )
        logger.debug(f"Equilibrium [{i:5d}] time={tok.equilibrium.time_slice.current.time:.2f}s")

        tok.advance()

    # convert -delay 10 -loop 1 tok_east*.png tok_east_70754.mp4

    logger.debug(tok.equilibrium.time_slice.time)
