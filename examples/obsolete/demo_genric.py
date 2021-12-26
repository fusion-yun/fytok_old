from typing import TypeVar,  Sequence
import collections
from spdm.data import Dict
from spdm.common.logger import logger
if __name__ == "__main__":

    cache = {}

    g = Dict[str, dict](cache)

    logger.debug(getattr(g, "__orig_class__", None))

    g["a"] = "Hello world!"
    g["b"] = {"c": 2, "d": 3}

    logger.debug(cache)
    logger.debug(type(g["b"]))
