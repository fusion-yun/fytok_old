
import collections

import numpy as np
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreSources import CoreSources
from fytok.modules.transport.Equilibrium import Equilibrium
from spdm.logger import logger
from spdm.data import Dict, List, Node
from spdm.data import Function


class CoreSourceDummy(CoreSources.Source):
    def __init__(self, d=None, *args,  **kwargs):

        super().__init__(collections.ChainMap({
            "identifier": {"name": "unspecified", "index": 5,
                           "description": f"{self.__class__.__name__} Dummy CoreTransport.Model "},
            "code": {"name": "dummy"}}, d or {}),
            *args, **kwargs)

    def refresh(self, *args,  **kwargs) -> float:
        return super().refresh(*args, **kwargs)


__SP_EXPORT__ = CoreSourceDummy
