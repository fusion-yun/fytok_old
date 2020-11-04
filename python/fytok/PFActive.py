import collections
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.LazyProxy import LazyProxy


class Coil(AttributeTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PFActive(AttributeTree):

    def __init__(self, *args,  **kwargs):
        super().__init__()
        if len(args)+len(kwargs) > 0:
            self.load(*args, **kwargs)

    def load(self, ids=None,  *args, coils=None, circuit=None, supply=None, **kwags):
        if isinstance(ids, LazyProxy):
            if coils is None:
                coils = ids.coil

            if circuit is None:
                circuit = ids.circuit

            if supply is None:
                supply = ids.supply

        if coils is None:
            pass
        elif isinstance(coils, LazyProxy):
            for coil in coils:
                if coil.element.geometry.geometry_type != 2:
                    raise NotImplementedError()
                rect = coil.element.geometry.rectangle
                next_coil = self.entry.coil.__push_back__()
                next_coil.name = str(coil.name)
                next_coil.r = float(rect.r)
                next_coil.z = float(rect.z)
                next_coil.width = float(rect.width)
                next_coil.height = float(rect.height)
                next_coil.turns = int(coil.element[0].turns_with_sign)
        else:
            raise NotImplementedError()
        return self.entry

    @property
    def coil(self):
        return LazyProxy(super().__getitem__("coil"))

    @property
    def circuit(self):
        return LazyProxy(super().__getitem__("circuit"))

    @property
    def supply(self):
        return LazyProxy(super().__getitem__("supply"))

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
