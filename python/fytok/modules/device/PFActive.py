
import matplotlib.pyplot as plt
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Node import Dict, List
from spdm.data.Node import sp_property
from spdm.util.logger import logger
import collections
from ..common.IDS import IDS


class PFActiveCoil(Dict):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def name(self) -> str:
        return self["name"]

    @sp_property
    def identifier(self) -> str:
        return self["identifier"]

    @sp_property
    def element(self) -> List[AttributeTree]:
        return self["element"]


class PFActiveCircuit(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PFActiveSupply(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PFActive(IDS):
    """
    """
    _IDS = "pf_active"

    Coil = PFActiveCoil
    Circuit = PFActiveCircuit
    Supply = PFActiveSupply

    def __init__(self, *args,  **kwargs):
        super().__init__(*args,  **kwargs)

    @sp_property
    def coil(self) -> List[Coil]:
        return self["coil"]

    @sp_property
    def circuit(self) -> List[Circuit]:
        """Circuits, connecting multiple PF coils to multiple supplies,
            defining the current and voltage relationships in the system"""
        return self["circuit"]

    @sp_property
    def supply(self) -> List[Supply]:
        """PF power supplies"""
        return self["supply"]

    def plot(self, axis=None, *args, with_circuit=False, **kwargs):

        if axis is None:
            axis = plt.gca()
        for coil in self.coil:
            rect = coil.element[0].geometry.rectangle
            axis.add_patch(plt.Rectangle((rect.r - rect.width / 2.0,  rect.z - rect.height / 2.0),
                                         rect.width,  rect.height,
                                         **collections.ChainMap(kwargs,  {"fill": False})))
            axis.text(rect.r, rect.z, coil.name,
                      horizontalalignment='center',
                      verticalalignment='center')

        return axis
