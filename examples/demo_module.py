import pathlib

from fytok.transport.Equilibrium import Equilibrium
from spdm.util.logger import logger


if __name__ == "__main__":
    logger.info("====== START ========")
    output_path = pathlib.Path('/home/salmon/workspace/output')

    mod = Equilibrium({"code": {"name": "dummy"}})

    logger.debug(type(mod))
