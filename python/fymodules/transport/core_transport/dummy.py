
import collections

import numpy as np
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreTransport import CoreTransport
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.data import Dict, List, Node
from spdm.common.logger import logger


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
