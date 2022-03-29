
import collections

import numpy as np
from fytok.modules.transport.CoreProfiles import CoreProfiles
from fytok.modules.transport.CoreTransport import CoreTransport
from fytok.modules.transport.Equilibrium import Equilibrium
from spdm.data import Function
from spdm.data import Dict, List, Node
from spdm.logger import logger


class TransportModelDummy(CoreTransport.Model):
    """
       Dummy CoreTransport.Model
       ===============================

    """

    def __init__(self, d, *args,   **kwargs):
        super().__init__(collections.ChainMap({
            "identifier": {"name": "dummy", "index": 5,
                           "description": f"{self.__class__.__name__} Dummy CoreTransport.Model "},
            "code": {"name": "spitzer"}}, d or {}),
            *args, **kwargs)

    def refresh(self, *args, **kwargs) -> float:
        return super().refresh(*args, **kwargs)


__SP_EXPORT__ = TransportModelDummy
