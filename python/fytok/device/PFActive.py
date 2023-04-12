
import collections
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.Dict import Dict
from spdm.data.List import List
from spdm.data.TimeSeries import TimeSeries
from spdm.data.sp_property import sp_property

from ..common.IDS import IDS
from ..common.Misc import RZTuple


class PFActiveCoil(Dict):

    name: str = sp_property()
    identifier: str = sp_property()

    class Element(Dict):

        name: str = sp_property()
        """Name of this element {static}	STR_0D	"""

        identifier: str = sp_property()
        """Identifier of this element {static}	STR_0D	"""

        turns_with_sign: float = sp_property(default_value=1)
        """Number of effective turns in the element for calculating magnetic fields of the coil/loop;
            includes the sign of the number of turns (positive means current is counter-clockwise when seen from above) {static} [-]	FLT_0D	"""

        area: float = sp_property(default_value=0.0)
        """Cross-sectional areas of the element {static} [m^2]	FLT_0D    """

        class Geometry(Dict):

            geometry_type: int = sp_property(default_value=2)
            """Type used to describe the element shape (1:'outline', 2:'rectangle', 3:'oblique', 4:'arcs of circle') {static}	INT_0D	"""

            outline: RZTuple = sp_property(default_value={})
            """Irregular outline of the element. Do NOT repeat the first point.	structure	"""

            @dataclass
            class Rectangle:
                r: float = 0
                z: float = 0
                width: float = 0
                height: float = 0

            rectangle: Rectangle = sp_property()
            """Rectangular description of the element	structure	"""

            @dataclass
            class Oblique:
                r: float
                z: float
                length: float = 0.0
                thickness: float = 0.0
                beta: float = 0.0

            oblique: Oblique = sp_property()
            """Trapezoidal description of the element	structure	"""

            @dataclass
            class ArcsOfCircle(Dict):
                r: np.ndarray
                z: np.ndarray
                curvature_radii: np.ndarray

            arcs_of_circle: ArcsOfCircle = sp_property()
            """
                Description of the element contour by a set of arcs of circle. For each of these, the position of the start point is given together
                with the curvature radius. The end point is given by the start point of the next arc of circle.
            """

        geometry: Geometry = sp_property()

    element: List[Element] = sp_property()

    current: TimeSeries = sp_property()

    voltage: TimeSeries = sp_property()


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

    circuit: List[Coil] = sp_property()
    """Circuits, connecting multiple PF coils to multiple supplies,
            defining the current and voltage relationships in the system"""

    supply: List[Coil] = sp_property()
    """PF power supplies"""

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
                      verticalalignment='center',
                      fontsize='xx-small')

        return axis
