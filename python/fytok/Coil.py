import collections
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.LazyProxy import LazyProxy
from spdm.util.urilib import urisplit


class Coil(AttributeTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Circuit(AttributeTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
