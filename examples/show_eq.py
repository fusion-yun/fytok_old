import pathlib
import os
import numpy as np
from spdm.data.File import File
from spdm.view.View import display
from spdm.utils.logger import logger
from fytok.modules.Equilibrium import Equilibrium
from fytok.Tokamak import Tokamak

WORKSPACE = "/home/salmon/workspace"  # "/ssd01/salmon_work/workspace/"

os.environ["SP_DATA_MAPPING_PATH"] = f"{WORKSPACE}/fytok_data/mapping"


if __name__ == "__main__":
    output_path = pathlib.Path(f"{WORKSPACE}/output/")

    eq = Equilibrium(f"file+GEQdsk://{WORKSPACE}/gacode/neo/tools/input/profile_data/g141459.03890")

    # logger.debug(eq.time_slice[0].profiles_2d[0].psi.__value__)

    display(eq, title=f"EQUILIBRIUM", output=output_path/"EQUILIBRIUM.svg")

    tok = Tokamak(f"EAST+MDSplus://{WORKSPACE}/fytok_data/mdsplus/~t/?shot=70745")

    tok.equilibrium.refresh(time=5.0)

    display(tok, title=f"EAST", output=output_path / "tok_east.svg")
