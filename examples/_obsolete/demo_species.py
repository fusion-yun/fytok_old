

from fytok.transport.Species import Species
from spdm.utils.logger import logger


if __name__ == "__main__":
    sp = Species({"label": "H"})
    logger.debug(sp.label)
    logger.debug(sp.element)