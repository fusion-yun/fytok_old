import pathlib

import numpy as np
from spdm.utils.logger import logger
from spdm.data.File import File
from spdm.data.Entry import Entry


def mhdin_outline(r, z, w, h, a0, a1):
    return r, z, r+w, z+h


def sp_to_imas(data: dict):

    entry = Entry({})
    logger.debug(data.keys())

    entry["wall.description_2d[0].limiter.unit[0].outline.r"] = np.array(data["rsi"])
    entry["wall.description_2d[0].limiter.unit[0].outline.z"] = np.array(data["zsi"])
    logger.debug(data["rsi"])
    vessel_in_r, vessel_in_z, vessel_out_r, vessel_out_z = mhdin_outline(
        data["rvs"], data["zvs"], data["wvs"], data["hvs"], data["avs"], data["avs2"]
    )

    entry["wall.description_2d[0].vessel.unit[0].annular.outline_inner.r"] = np.array(vessel_in_r)
    entry["wall.description_2d[0].vessel.unit[0].annular.outline_inner.z"] = np.array(vessel_in_z)
    entry["wall.description_2d[0].vessel.unit[0].annular.outline_outer.r"] = np.array(vessel_out_r)
    entry["wall.description_2d[0].vessel.unit[0].annular.outline_outer.z"] = np.array(vessel_out_z)

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

        return sp_to_imas(data["in3"])
