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

    entry = open_entry(f"file+mhdin://{WORKSPACE}/omas/omas/machine_mappings/support_files/d3d/181292/mhdin.dat")

    with File(f"/home/salmon/workspace/fytok_data/mapping/d3d/imas/3/d3d.xml", mode="w", root="mapping", format="xml") as f:
        f.write(entry.dump())

    # logger.debug(entry.get("wall.description_2d[0].limiter.unit[0].r"))

    tok = Tokamak("d3d")

    # logger.debug(entry.get("equilibrium/time_slice[0]/vacuum_toroidal_field/r0"))

    # tok = Tokamak(entry=[
    #     f"file+iterdb://{WORKSPACE}/gacode/neo/tools/input/profile_data/iterdb141459.03890",
    #     # f"file+geqdsk://{WORKSPACE}/gacode/neo/tools/input/profile_data/g141459.03890"
    # ],
    #     device="d3d",)

    # tok = Tokamak(f"file+iterdb://{WORKSPACE}/gacode/neo/tools/input/profile_data/iterdb141459.03890",
    #               device="D3D",
    #               equilibrium={"code": {"name": "eq_analyze"},
    #                            "$entry": [f"file+geqdsk://{WORKSPACE}/gacode/neo/tools/input/profile_data/g141459.03890#equilibrium"]
    #                            }
    #               )
    # tok = Tokamak(device="D3D",
    #               entry=[
    #                   f"file+iterdb://{WORKSPACE}/gacode/neo/tools/input/profile_data/iterdb141459.03890",
    #                   f"file+geqdsk://{WORKSPACE}/gacode/neo/tools/input/profile_data/g141459.03890"
    #               ],
    #               equilibrium={"code": {"name": "eq_analyze"}})

    display(tok.wall, title=f"{tok.device.upper()} RZ   View ", output=output_path/f"{tok.device}_rz.svg")
    # display(tok, title=f"{tok.device.upper()} Top  View ", output=output_path/"east_top.svg", view="TOP")
