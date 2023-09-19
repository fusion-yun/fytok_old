import os
import pathlib

import numpy as np
from spdm.data.Entry import Entry
from spdm.data.File import File
from spdm.utils.logger import logger
from spdm.view.View import display

from fytok.modules.Equilibrium import Equilibrium
from fytok.Tokamak import Tokamak

WORKSPACE = "/home/salmon/workspace"  # "/ssd01/salmon_work/workspace/"

os.environ["SP_DATA_MAPPING_PATH"] = f"{WORKSPACE}/fytok_data/mapping"


if __name__ == "__main__":
    output_path = pathlib.Path(f"{WORKSPACE}/output/")

    ############################################################################
    # Example: gfile I/O,  equilibrium.dump()

    eq0 = Equilibrium(f"file+GEQdsk://{WORKSPACE}/gacode/neo/tools/input/profile_data/g141459.03890#equilibrium")

    display(eq0, title=f"EQUILIBRIUM", output=output_path/"EQUILIBRIUM.svg")

    with File(f"{output_path}/EQUILIBRIUM.gfile", mode="w", format="GEQdsk") as oid:
        oid.write({"equilibrium": eq0.dump()})

    eq1 = Equilibrium(f"file+GEQdsk://{WORKSPACE}/output/EQUILIBRIUM.gfile#equilibrium")

    logger.debug(eq0.time_slice.current.profiles_1d.q.__value__)

    logger.debug(eq1.time_slice.current.profiles_1d.q.__value__)

    ##############################################################################

    shot = 70745

    tok = Tokamak(f"EAST+MDSplus://{WORKSPACE}/fytok_data/mdsplus/~t/?enable=efit_east&shot={shot}")

    tok.refresh(time=2.0)

    psi = tok.equilibrium.time_slice.current.profiles_2d[0].psi.__value__

    psi_min = psi.min()

    psi_max = psi.max()

    levels = np.arange(psi_min, psi_max, (psi_max-psi_min)/40)

    for i in range(150):
        display(tok.equilibrium,
                title=f"EAST shot={shot} time={tok.equilibrium.time_slice.current.time:.2f}s ",
                output=output_path / f"tok_east_{int(tok.equilibrium.time_slice.current.time*100)}.png",
                transparent=False,
                psi={"$matplotlib": {"levels": levels}}
                )
        logger.debug(f"Equilibrium [{i:5d}] time={tok.equilibrium.time_slice.current.time:.2f}s")
        tok.advance()

    # convert -delay 10 -loop 1 tok_east*.png tok_east_70754.mp4

    logger.debug(tok.equilibrium.time_slice.time)
