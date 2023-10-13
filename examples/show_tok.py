import os
import pathlib

from fytok.Tokamak import Tokamak
from fytok.utils.logger import logger
from spdm.view.View import display

WORKSPACE = "/home/salmon/workspace"  # "/ssd01/salmon_work/workspace/"

if __name__ == "__main__":

    output_path = pathlib.Path(f"{WORKSPACE}/output")

    tok = Tokamak(
        # f"file+GEQdsk://{WORKSPACE}/gacode/neo/tools/input/profile_data/g141459.03890",
        # device="d3d"
        f"mdsplus://{WORKSPACE}/fytok_data/mdsplus/~t/?enable=efit_east",
        device="EAST", shot=70754,
    )
    # desc = tok.wall.description_2d[0]
    # logger.debug(desc.vessel.unit)
    # logger.debug(desc.vessel.unit[0].annular.outline_outer.r)

    tok.refresh(time=5.0)

    # logger.debug(tok.equilibrium.time_slice.current.vacuum_toroidal_field.b0)

    # profiles_1d = tok.equilibrium.time_slice.current.profiles_1d

    # profiles_2d = tok.equilibrium.time_slice.current.profiles_2d

    # coordinate_system = tok.equilibrium.time_slice.current.coordinate_system

    # logger.debug(profiles_1d.f_df_dpsi.__array__())

    # display([tok.equilibrium],
    #         title=tok.short_description,  output=output_path/f"{tok.tag}_rz.svg")
    display(tok, title=tok.short_description,
            styles={"interferometer": False},
            output=output_path/f"{tok.tag}_rz.svg")

    # display(tok, title=tok.short_description,   output=output_path/f"{tok.tag}_top.svg", view_point="TOP")
