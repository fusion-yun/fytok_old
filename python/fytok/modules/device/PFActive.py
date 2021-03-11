import collections
from copy import copy
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.PhysicalGraph import PhysicalGraph, _next_
from spdm.util.logger import logger


class PFActive(PhysicalGraph):
    """
    """

    class Coil(PhysicalGraph):
        def __init__(self,  *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Circuit(PhysicalGraph):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Supply(PhysicalGraph):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def __init__(self, *args,  **kwargs):
        super().__init__(*args,  **kwargs)

    @cached_property
    def coil(self):
        return PFActive.Coil(self["coil"], parent=self)

    @cached_property
    def circuit(self):
        """Circuits, connecting multiple PF coils to multiple supplies, 
            defining the current and voltage relationships in the system"""
        return PFActive.Circuit(self["circuit"], parent=self)

    @cached_property
    def supply(self):
        """PF power supplies"""
        return PFActive.Supply(self["supply"], parent=self)

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
