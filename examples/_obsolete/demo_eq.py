import pathlib
import pandas as pd
import numpy as np
from fytok.modules.Equilibrium import Equilibrium
from fytok.utils.plot_profiles import sp_figure, plot_profiles
from spdm.data.Function import function_like
from spdm.data.File import File
from spdm.utils.logger import logger

if __name__ == "__main__":

    logger.info("====== START ========")
    output_path = pathlib.Path('/home/salmon/workspace/output')

    ###################################################################################################
    # baseline
    device_desc = File("/home/salmon/workspace/fytok_data/mapping/ITER/imas/3/static/config.xml", format="XML").read()
    # Equilibrium
    eqdsk_file = File(
        "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt", format="GEQdsk").read()

    eq = Equilibrium({**eqdsk_file.dump(), "code": {"name":  "eq_analyze"}},
                     default_value={"time_slice": {"coordinate_system": {"grid": {"dim1": 128, "dim2": 64}}}})

    logger.debug(eq.time_slice[-1].global_quantities.magnetic_axis.r)
