import collections
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from spdm.util.logger import logger


class PFCoils:
    Coil = collections.namedtuple("Coil", "r z width height turns")

    def __init__(self, *args, **kwargs):
        self._coils = {}

    def __iter__(self):
        for name, coil in self._coils.items():
            yield name, coil

    def add(self, name, **kwargs):
        self._coils[name] = PFCoils.Coil(**kwargs)

    def plot(self, axis=None, **kwargs):

        if axis is None:
            axis = plt.gca()

        for coil in self._coils.values():
            axis.add_patch(plt.Rectangle((coil.r-coil.width/2.0, coil.z-coil.height/2.0),
                                         coil.width, coil.height, fill=False), **kwargs)

        return axis
