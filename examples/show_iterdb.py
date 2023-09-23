import os
import pathlib

import numpy as np
from fytok.Tokamak import Tokamak
from fytok.utils.logger import logger
from spdm.data.File import File
from spdm.data.Entry import Entry, open_entry
from spdm.view.View import display

WORKSPACE = "/home/salmon/workspace"  # "/ssd01/salmon_work/workspace/"

os.environ["SP_DATA_MAPPING_PATH"] = f"{WORKSPACE}/fytok_data/mapping"

if __name__ == "__main__":

    output_path = pathlib.Path(f"{WORKSPACE}/output")

    # entry = open_entry(f"EAST+file+iterdb://{WORKSPACE}/gacode/neo/tools/input/profile_data/iterdb141459.03890")

    # logger.debug(entry.get("equilibrium/time_slice[0]/vacuum_toroidal_field/r0"))

    # tok = Tokamak(f"D3D+file+iterdb://{WORKSPACE}/gacode/neo/tools/input/profile_data/iterdb141459.03890",
    #               equilibrium={"code": {"name": "eq_analyze"},
    #                            "$entry": [f"file+geqdsk://{WORKSPACE}/gacode/neo/tools/input/profile_data/g141459.03890#equilibrium"]
    #                            }
    #               )
    tok = Tokamak(device="D3D",
                  entry=[
                      f"file+iterdb://{WORKSPACE}/gacode/neo/tools/input/profile_data/iterdb141459.03890",
                      f"file+geqdsk://{WORKSPACE}/gacode/neo/tools/input/profile_data/g141459.03890"
                  ],
                  equilibrium={"code": {"name": "eq_analyze"}})

    display(tok, title=f"{tok.device.upper()} RZ   View ", output=output_path/"east_rz.svg")
    display(tok, title=f"{tok.device.upper()} Top  View ", output=output_path/"east_top.svg", view="TOP")
