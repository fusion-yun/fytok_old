import pathlib
import os
import numpy as np
from spdm.data.File import File
from spdm.view.View import display
from spdm.utils.logger import logger
from fytok.modules.Equilibrium import Equilibrium
from fytok.Tokamak import Tokamak

DATA_PATH = "/home/salmon/workspace/fytok_data"

# GACODE_PATH=os.environ['GACODE_PATH']
SP_OUTPUT = os.environ.get('SP_OUTPUT', '../output')

if __name__ == "__main__":

    output_path = pathlib.Path(SP_OUTPUT)

    tok = Tokamak(f"file+MDSplus[EAST]://{DATA_PATH}/mdsplus/~t/?tree_name=efit_east#70745")

    display(tok, title=f"Tokamak", output=output_path/"tok.svg")
    for coil in tok.pf_active.coil:
        current=coil.current
        logger.debug(current)
    logger.info("Done")
