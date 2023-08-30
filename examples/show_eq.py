import pathlib

import numpy as np
from spdm.data.File import File
from spdm.view.View import display
from spdm.utils.logger import logger
from fytok.modules.Equilibrium import Equilibrium

if __name__ == "__main__":

    output_path = pathlib.Path('/home/salmon/workspace/output')

    eq = Equilibrium("GEQdsk:///home/salmon/workspace/gacode/neo/tools/input/profile_data/g141459.03890")

    display(eq, title=f"EQUILIBRIUM", output=output_path/"EQUILIBRIUM.svg")

    logger.info("Done")
