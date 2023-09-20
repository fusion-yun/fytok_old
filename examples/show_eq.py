import os
import pathlib

import numpy as np
from fytok.modules.Equilibrium import Equilibrium
from fytok.Tokamak import Tokamak
from fytok.utils.logger import logger
from spdm.data.Entry import Entry
from spdm.data.File import File
from spdm.view.View import display

WORKSPACE = "/home/salmon/workspace"  # "/ssd01/salmon_work/workspace/"

os.environ["SP_DATA_MAPPING_PATH"] = f"{WORKSPACE}/fytok_data/mapping"
output_path = pathlib.Path(f"{WORKSPACE}/output/")


if __name__ == "__main__":

    # Example: gfile I/O,  equilibrium.dump()

    eq0 = Equilibrium(f"file+GEQdsk://{WORKSPACE}/gacode/neo/tools/input/profile_data/g141459.03890#equilibrium")

    display(eq0, title=f"EQUILIBRIUM", output=output_path/"EQUILIBRIUM.svg")

    with File(f"{output_path}/EQUILIBRIUM.gfile", mode="w", format="GEQdsk") as oid:
        oid.write({"equilibrium": eq0.dump()})

    eq1 = Equilibrium(f"file+GEQdsk://{WORKSPACE}/output/EQUILIBRIUM.gfile#equilibrium")

    logger.debug(eq0.time_slice.current.profiles_1d.q.__value__)

    logger.debug(eq1.time_slice.current.profiles_1d.q.__value__)
