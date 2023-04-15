import pathlib

import numpy as np
import pandas as pd
from fytok.transport.CoreTransport import CoreTransportModel
from fytok.transport.Equilibrium import Equilibrium
from spdm.util.logger import logger
import fymodules.core_transport.model.chang_hinton as chang
from spdm.data.plugins.PluginGEQdsk import GEQdskFile


if __name__ == "__main__":
    logger.info("====== START ========")
    output_path = pathlib.Path('/home/salmon/workspace/output')

    mod = Equilibrium({"code": {"name": "dummy"}})

    logger.debug(type(mod))
