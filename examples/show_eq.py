import pathlib
import os
import numpy as np
from spdm.data.File import File
from spdm.view.View import display
from spdm.utils.logger import logger
from fytok.modules.Equilibrium import Equilibrium

GACODE_PATH=os.environ['GACODE_PATH']
SP_OUTPUT=os.environ['SP_OUTPUT']

if __name__ == "__main__":

    output_path = pathlib.Path(SP_OUTPUT)

    eq = Equilibrium(f"GEQdsk://{GACODE_PATH}/neo/tools/input/profile_data/g141459.03890")

    display(eq, title=f"EQUILIBRIUM", output=output_path/"EQUILIBRIUM.svg")

    logger.info("Done")
