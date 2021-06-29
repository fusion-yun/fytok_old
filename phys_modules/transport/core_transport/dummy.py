
import collections

from spdm.numlib import np
from spdm.numlib import constants
from fytok.transport.CoreProfiles import CoreProfiles
from fytok.transport.CoreTransport import CoreTransport
from fytok.transport.Equilibrium import Equilibrium
from spdm.data.Function import Function
from spdm.data.Node import Dict, List, Node
from spdm.util.logger import logger


class TransportModelDummy(CoreTransport.Model):
    """
       Dummy CoreTransport.Model
       ===============================

    """

    def __init__(self, d, /,   **kwargs):
        super().__init__(d,
                         identifier={
                             "name": "dummy",
                             "index": 5,
                             "description": f"{self.__class__.__name__} Dummy CoreTransport.Model "
                         }, **kwargs)

    def refresh(self, *args, **kwargs) -> None:
        logger.debug(f"Dummy actor: Nothing to do!")


__SP_EXPORT__ = TransportModelDummy
