import collections
from copy import copy
from functools import cached_property

import matplotlib.pyplot as plt
from spdm.data.AttributeTree import as_attribute_tree
from spdm.data.Node import Dict
from spdm.util.logger import logger
from ..utilities.IDS import IDS


@as_attribute_tree
class PFActive(IDS):
    """
    """

    class Coil(Dict):
        def __init__(self,  *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Circuit(Dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Supply(Dict):
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
            geo = coil.element.geometry.rectangle
            axis.add_patch(plt.Rectangle((geo.r-geo.width/2.0,  geo.z-geo.height/2.0),
                                         geo.width,  geo.height,
                                         **collections.ChainMap(kwargs,  {"fill": False})))
            axis.text(geo.r, geo.z, str(coil.name),
                      horizontalalignment='center',
                      verticalalignment='center')

        return axis
