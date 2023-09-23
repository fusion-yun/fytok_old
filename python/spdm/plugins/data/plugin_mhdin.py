import pathlib

import numpy as np
from spdm.utils.logger import logger
from spdm.data.File import File
from spdm.data.Entry import Entry


def sp_to_imas(data: dict):
    logger.debug(data)
    entry = Entry({})

    return entry


@File.register(["mhdin"])
class MHDINFile(File):
    """ READ mahchine description file (MHDIN)
        learn from omas
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read(self) -> Entry:
        if self.url.authority:
            raise NotImplementedError(f"{self.url}")

        data = File(self.url.path, mode="r", format="namelist").read().dump()

        return sp_to_imas(data)
