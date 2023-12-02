import collections
import numpy as np
from spdm.data.sp_property import sp_tree
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreSources import CoreSources
from fytok.modules.Equilibrium import Equilibrium
from fytok.utils.logger import logger


@sp_tree
class CoreSourceDummy(CoreSources.Source):
    code = {"name": "dummy", "description": " Dummy CoreSources.Source "}

    identifier = "unspecified"


__SP_EXPORT__ = CoreSourceDummy
