
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger


class CoreTransports(AttributeTree):
    def __init__(self, config=None, *args, tokamak=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.load(config)

    def load(self, *args, **kwargs):
        logger.debug("NOT IMPLEMENTED")

    def update(self, *args, **kwargs):
        logger.debug("NOT IMPLEMENTED")

    @property
    def mode(self):
        logger.debug("NOT IMPLEMENTED")

        yield None
