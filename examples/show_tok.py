import os
import pathlib

from fytok.Tokamak import Tokamak
from fytok.utils.logger import logger
from spdm.view.View import display

WORKSPACE = "/home/salmon/workspace"  # "/ssd01/salmon_work/workspace/"

if __name__ == "__main__":

    output_path = pathlib.Path(f"{WORKSPACE}/output")

    tok = Tokamak(f"mdsplus://{WORKSPACE}/fytok_data/mdsplus/~t/?enable=efit_east", device="EAST", shot=70754,)
    # f"file+GEQdsk://{WORKSPACE}/gacode/neo/tools/input/profile_data/g141459.03890", device="d3d"

    tok.refresh({"time": 5})

    display(tok,
            title=tok.short_description,
            styles={"interferometer": False},
            output=output_path/f"{tok.tag}_rz.svg")

    tok.advance()

    display(tok,
            title=tok.short_description,
            styles={"interferometer": False},
            output=output_path/f"{tok.tag}_rz.svg")
    tok.advance()

    display(tok,
            title=tok.short_description,
            styles={"interferometer": False},
            output=output_path/f"{tok.tag}_rz.svg")
