import collections
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.LazyProxy import LazyProxy
from spdm.util.urilib import urisplit


from .Coil import Coil


class PFActive(AttributeTree):

    def __init__(self, config, *args,  **kwargs):
        super().__init__(*args,  **kwargs)
        self.load(config)

    def load(self, config):

        if isinstance(config, LazyProxy):
            for coil in config.coil:
                if coil.element.geometry.geometry_type != 2:
                    raise NotImplementedError()
                rect = coil.element.geometry.rectangle

                _, next_coil = self.coil.__push_back__()
                next_coil.name = str(coil.name)
                next_coil.r = float(rect.r)
                next_coil.z = float(rect.z)
                next_coil.width = float(rect.width)
                next_coil.height = float(rect.height)
                next_coil.turns = int(coil.element[0].turns_with_sign)
        else:
            raise NotImplementedError()

    def plot(self, axis=None, *args, with_circuit=False, **kwargs):

        if axis is None:
            axis = plt.gca()

        for coil in self.coil:
            axis.add_patch(
                plt.Rectangle(
                    (coil.r-coil.width/2.0,
                     coil.z-coil.height/2.0),
                    coil.width,
                    coil.height,
                    fill=False))

        return axis
