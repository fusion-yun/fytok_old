
import collections
from dataclasses import dataclass

import matplotlib.pyplot as plt
from fytok.common.Misc import Identifier, RZTuple, Signal
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Node import Dict, List, sp_property
from spdm.numlib import np
from spdm.util.logger import logger

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

    class Element(Dict):
        def __init__(self,   *args, **kwargs):
            super().__init__(*args, **kwargs)

        @sp_property
        def name(self) -> str:
            """Name of this element {static}	STR_0D	"""
            return self.get("name", 0.0)

        @sp_property
        def identifier(self) -> Identifier:
            """Identifier of this element {static}	STR_0D	"""
            return self.get("identifier", " 0.0")

        @sp_property
        def turns_with_sign(self) -> str:
            """Number of effective turns in the element for calculating magnetic fields of the coil/loop; 
            includes the sign of the number of turns (positive means current is counter-clockwise when seen from above) {static} [-]	FLT_0D	"""
            return self.get("turns_with_sign", 1)

        @sp_property
        def area(self) -> str:
            """Cross-sectional areas of the element {static} [m^2]	FLT_0D    """
            return self.get("area", 0.0)

        class Geometry(Dict):
            @sp_property
            def geometry_type(self) -> int:
                """Type used to describe the element shape (1:'outline', 2:'rectangle', 3:'oblique', 4:'arcs of circle') {static}	INT_0D	"""
                return self.get("geometry_type", None)

            @sp_property
            def outline(self) -> RZTuple:
                """Irregular outline of the element. Do NOT repeat the first point.	structure	"""
                return RZTuple(**self.get("outline", {}))

            @dataclass
            class Rectangle(Dict):
                r: float = 0
                z: float = 0
                width: float = 0
                height: float = 0

            @sp_property
            def rectangle(self) -> Rectangle:
                """Rectangular description of the element	structure	"""
                return self.get("rectangle", {})

            @dataclass
            class Oblique(Dict):
                r: float
                z: float
                length: float = 0.0
                thickness: float = 0.0
                beta: float = 0.0

            @sp_property
            def oblique(self) -> Oblique:
                """Trapezoidal description of the element	structure	"""
                return self.get("oblique", {})

            @dataclass
            class ArcsOfCircle(Dict):
                r: np.ndarray
                z: np.ndarray
                curvature_radii: np.ndarray

            @sp_property
            def arcs_of_circle(self) -> ArcsOfCircle:
                """
                    Description of the element contour by a set of arcs of circle. For each of these, the position of the start point is given together 
                    with the curvature radius. The end point is given by the start point of the next arc of circle.
                """
                return self.get("", {})

        @sp_property
        def geometry(self) -> Geometry:
            return self.get("geometry", {})

    @sp_property
    def element(self) -> List[Element]:
        return self.get("element", [])

    @sp_property
    def current(self) -> Signal:
        return self.get("current", None)

    @sp_property
    def voltage(self) -> Signal:
        return self.get("voltage", None)


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

    @ sp_property
    def coil(self) -> List[Coil]:
        return self["coil"]

    @ sp_property
    def circuit(self) -> List[Circuit]:
        """Circuits, connecting multiple PF coils to multiple supplies,
            defining the current and voltage relationships in the system"""
        return self["circuit"]

    @ sp_property
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
