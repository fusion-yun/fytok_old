
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger


class CoreTransport(AttributeTree):
    r"""Core Transport
    
    Todo: 
        * transport
        * need complete

    """
    IDS="core_transport"

    def __init__(self, config=None, *args, tokamak=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.update(config, tokamak=tokamak)

    def update(self, *args, tokamak=None, **kwargs):
        if tokamak is not None:
            self.__dict__["_tokamak"] = tokamak
        logger.debug("NOT IMPLEMENTED")


    @property
    def mode(self):
        logger.debug("NOT IMPLEMENTED")
        yield None
