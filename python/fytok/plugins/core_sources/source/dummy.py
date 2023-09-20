
import collections

import numpy as np
from fytok.modules.CoreProfiles import CoreProfiles
from fytok.modules.CoreSources import CoreSources
from fytok.modules.Equilibrium import Equilibrium
from fytok.utils.logger import logger


class CoreSourceDummy(CoreSources.Source):
    def __init__(self, d=None, *args,  **kwargs):

        super().__init__(collections.ChainMap({
            "identifier": {"name": "unspecified", "index": 5,
                           "description": f"{self.__class__.__name__} Dummy CoreTransport.Model "},
            "code": {"name": "dummy"}}, d or {}),
            *args, **kwargs)

    def refresh(self, *args,  **kwargs):
        return super().refresh(*args, **kwargs)


__SP_EXPORT__ = CoreSourceDummy
