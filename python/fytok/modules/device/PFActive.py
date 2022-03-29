
import collections
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from spdm.data import (Dict, File, Function, Link, List, Node, Path, Query,
                       sp_property)
from spdm.logger import logger

from ...IDS import IDS
from ..common.Misc import RZTuple, Signal


class PFActiveCoil(Dict):

    name: str = sp_property()
    identifier: str = sp_property()

    class Element(Dict):

        name: str = sp_property(doc="""Name of this element {static}	STR_0D	""")

        identifier: str = sp_property(doc="""Identifier of this element {static}	STR_0D	""")

        turns_with_sign: float = sp_property(doc="""Number of effective turns in the element for calculating magnetic fields of the coil/loop;
            includes the sign of the number of turns (positive means current is counter-clockwise when seen from above) {static} [-]	FLT_0D	""", default_value=1)

        area: float = sp_property(
            doc="""Cross-sectional areas of the element {static} [m^2]	FLT_0D    """, default_value=0.0)

        class Geometry(Dict):

            geometry_type: int = sp_property(
                doc="""Type used to describe the element shape (1:'outline', 2:'rectangle', 3:'oblique', 4:'arcs of circle') {static}	INT_0D	""", default_value=2)

            outline: RZTuple = sp_property(doc="""Irregular outline of the element. Do NOT repeat the first point.	structure	""",
                                           default_value={})

            @dataclass
            class Rectangle:
                r: float = 0
                z: float = 0
                width: float = 0
                height: float = 0

            rectangle: Rectangle = sp_property(doc="""Rectangular description of the element	structure	""")

            @dataclass
            class Oblique:
                r: float
                z: float
                length: float = 0.0
                thickness: float = 0.0
                beta: float = 0.0

            oblique: Oblique = sp_property(doc="""Trapezoidal description of the element	structure	""")

            @dataclass
            class ArcsOfCircle(Dict):
                r: np.ndarray
                z: np.ndarray
                curvature_radii: np.ndarray

            arcs_of_circle: ArcsOfCircle = sp_property(doc="""
                    Description of the element contour by a set of arcs of circle. For each of these, the position of the start point is given together
                    with the curvature radius. The end point is given by the start point of the next arc of circle.
                """)

        geometry: Geometry = sp_property()

    element: List[Element] = sp_property()

    current: Signal = sp_property()

    voltage: Signal = sp_property()


class PFActiveCircuit(Dict):
    pass


class PFActiveSupply(Dict):
    pass


class PFActive(IDS):
    """
    """
    _IDS = "pf_active"

    Coil = PFActiveCoil
    Circuit = PFActiveCircuit
    Supply = PFActiveSupply

    coil: List[Coil] = sp_property()

    circuit: List[Coil] = sp_property(doc="""Circuits, connecting multiple PF coils to multiple supplies,
            defining the current and voltage relationships in the system""")

    supply: List[Coil] = sp_property(doc="""PF power supplies""")

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
