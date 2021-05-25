
import matplotlib.pyplot as plt
from spdm.data.Node import Dict, List
from spdm.data.sp_property import sp_property
from spdm.util.logger import logger

from ..common.IDS import IDS


class PFActiveCoil(Dict):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)


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
            geo = coil["element.geometry.rectangle"]
            axis.add_patch(plt.Rectangle((geo["r"]-geo["width"]/2.0,  geo["z"]-geo["height"]/2.0),
                                         geo["width"],  geo["height"],
                                         **collections.ChainMap(kwargs,  {"fill": False})))
            axis.text(geo["r"], geo["z"], str(coil["name"]),
                      horizontalalignment='center',
                      verticalalignment='center')

        return axis
