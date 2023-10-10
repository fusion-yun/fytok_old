import os
import pathlib

from fytok.Tokamak import Tokamak
from fytok.utils.logger import logger
from spdm.view.View import display

WORKSPACE = "/home/salmon/workspace"  # "/ssd01/salmon_work/workspace/"

os.environ["SP_DATA_MAPPING_PATH"] = f"{WORKSPACE}/fytok_data/mapping"

if __name__ == "__main__":

    output_path = pathlib.Path(f"{WORKSPACE}/output")

    tok = Tokamak(
        # f"east+mdsplus://{WORKSPACE}/fytok_data/mdsplus/~t/?enable=efit_east&shot=70754",
        f"mdsplus://{WORKSPACE}/fytok_data/mdsplus/~t/?enable=efit_east",
        device="EAST", shot=70754,
    )

    tok.refresh(time=5.0)

    logger.debug(tok.equilibrium.time_slice.current.vacuum_toroidal_field.b0)

    profiles_1d = tok.equilibrium.time_slice.current.profiles_1d

    profiles_2d = tok.equilibrium.time_slice.current.profiles_2d

    # coordinate_system = tok.equilibrium.time_slice.current.coordinate_system

    logger.debug(profiles_1d.f_df_dpsi(profiles_1d.psi))

    # display([tok.equilibrium, tok.wall, tok.pf_active],
    #         title=tok.short_description,  output=output_path/f"{tok.tag}_rz.svg")
    display(tok, title=tok.short_description,
            styles={"interferometer": False},
            output=output_path/f"{tok.tag}_rz.svg")

    # display(tok, title=tok.short_description,   output=output_path/f"{tok.tag}_top.svg", view_point="TOP")
