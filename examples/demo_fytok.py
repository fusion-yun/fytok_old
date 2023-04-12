import pathlib
from copy import deepcopy

import numpy as np
import pandas as pd
from fytok.constants.Atoms import atoms
from fytok.device.Wall import Wall
from fytok.load_profiles import (load_core_profiles, load_core_source,
                                 load_core_transport, load_equilibrium)
from fytok.Tokamak import Tokamak
from scipy import constants
from spdm.data.File import File
from spdm.data.Function import Function, PiecewiseFunction
from spdm.util.logger import logger
from spdm.util.misc import get_many_value
from spdm.view.plot_profiles import plot_profiles, sp_figure

if __name__ == "__main__":
    logger.info("====== START ========")
    output_path = pathlib.Path('/home/salmon/workspace/output')

    ###################################################################################################
    # baseline
    device_desc = File("/home/salmon/workspace/fytokdata/mapping/ITER/imas/3/static/config.xml", format="XML").read()

    wall = Wall(device_desc.child("wall"))

    desc2d = wall.description_2d[0]

    logger.debug(desc2d.vessel.annular.outline_inner.r)

    logger.debug(device_desc[{"wall", "pf_active", "tf", "magnetics"}])
