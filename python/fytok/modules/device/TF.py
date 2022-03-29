
import collections
from copy import copy

import numpy as np
from ...IDS import IDS
from ..common.Signal import Signal
from spdm.logger import logger
from spdm.data import (Dict, File, Function, Link, List, Node, Path, Query,
                       sp_property)


class TF(IDS):
    """TF　Coils

    """
    _IDS = "tf"

    def __init__(self,  *args,    **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def r0(self) -> float:
        """Reference major radius of the device (from the official description of the device).

           This node is the placeholder for this official machine description quantity
            (typically the middle of the vessel at the equatorial midplane, although the exact
            definition may depend on the device) {static} [m]"""
        return self["r0"]

    @sp_property
    def is_periodic(self) -> bool:
        """Flag indicating whether coils are described one by one in the coil() structure (flag=0) or whether the coil
        structure represents only coils having different characteristics (flag = 1, n_coils must be filled in that case).
        In the latter case, the coil() sequence is repeated periodically around the torus. {static}	INT_0D	"""
        return self["is_periodic"]

    @sp_property
    def coil(self) -> List:
        """Set of coils around the tokamak {static}	struct_array [max_size=32]	1- 1...N"""
        return self["coil"]

    @sp_property
    def field_map(self) -> Dict:
        """Map of the vacuum field at various time slices, represented using the generic grid description {dynamic}
                struct_array [max_size=unbounded]	 """
        return self["field_map"]

    @sp_property
    def b_field_tor_vacuum_r(self) -> Signal:
        """Vacuum field times major radius in the toroidal field magnet. Positive sign means anti-clockwise when viewed 
            from above [T.m].  """
        return self["b_field_tor_vacuum_r"]

    @sp_property
    def delta_b_field_tor_vacuum_r(self) -> Signal:
        """Variation of (vacuum field times major radius in the toroidal field magnet) from the start of the plasma. [T.m]"""
        return self["delta_b_field_tor_vacuum_r"]
