import collections
from copy import copy
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from spdm.util.AttributeTree import AttributeTree, _next_
from spdm.util.logger import logger
from spdm.util.LazyProxy import LazyProxy
from spdm.util.urilib import urisplit


class PFActive(AttributeTree):
    """
    """

    def __init__(self, cache, *args,  **kwargs):
        super().__init__(*args,  **kwargs)
        self.__dict__['_cache'] = cache

    class Coil(AttributeTree):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Circuit(AttributeTree):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    @cached_property
    def coil(self):
        res = AttributeTree(default_factory_array=lambda _holder=self: PFActive.Coil(self))
        cache = self._cache.coil

        if isinstance(cache, LazyProxy):
            for coil in cache:
                if coil.element.geometry.geometry_type != 2:
                    raise NotImplementedError()
                rect = coil.element.geometry.rectangle

                next_coil = res[_next_]
                next_coil.name = str(coil.name)
                next_coil.r = float(rect.r)
                next_coil.z = float(rect.z)
                next_coil.width = float(rect.width)
                next_coil.height = float(rect.height)
                next_coil.turns = int(coil.element[0].turns_with_sign)
        else:
            if not isinstance(cache, AttributeTree):
                cache = AttributeTree(coil)

            for coil in cache:
                res[_next_] = coil

        return res

    @cached_property
    def circuit(self):
        """Circuits, connecting multiple PF coils to multiple supplies, 
            defining the current and voltage relationships in the system"""
        return NotImplemented

    @cached_property
    def supply(self):
        """PF power supplies"""
        return NotImplemented

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
