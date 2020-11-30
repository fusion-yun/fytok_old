
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger


class CoreTransport(AttributeTree):
    r"""Core Transport

    Todo: 
        * transport
        * need complete

    """
    IDS = "core_transport"

    def __init__(self, config=None, *args, tokamak=None, **kwargs):
        super().__init__(*args, **kwargs)
