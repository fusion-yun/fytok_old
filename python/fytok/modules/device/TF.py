
import collections
from copy import copy
from functools import cached_property, lru_cache

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.AttributeTree import as_attribute_tree
from spdm.data.Node import Dict
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.urilib import urisplit
from ..utilities.IDS import IDS

@as_attribute_tree
class TF(IDS):
    """TF　Coils

    """
    IDS = "tf"

    def __init__(self,  *args,    **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def r0(self):
        """Reference major radius of the device (from the official description of the device).

           This node is the placeholder for this official machine description quantity
            (typically the middle of the vessel at the equatorial midplane, although the exact
            definition may depend on the device) {static} [m]"""
        return self.cache.r0

    @cached_property
    def is_periodic(self):
        """Flag indicating whether coils are described one by one in the coil() structure (flag=0) or whether the coil
        structure represents only coils having different characteristics (flag = 1, n_coils must be filled in that case).
        In the latter case, the coil() sequence is repeated periodically around the torus. {static}	INT_0D	"""
        return self.cache.is_periodic

    @cached_property
    def coil(self):
        """Set of coils around the tokamak {static}	struct_array [max_size=32]	1- 1...N"""
        return NotImplemented

    @cached_property
    def field_map(self):
        """Map of the vacuum field at various time slices, represented using the generic grid description {dynamic}	struct_array [max_size=unbounded]	 """
        return NotImplemented

    @cached_property
    def b_field_tor_vacuum_r(self):
        """Vacuum field times major radius in the toroidal field magnet. Positive sign means anti-clockwise when viewed from above [T.m]. 
            This quantity is COCOS-dependent, with the following transformation :"""
        return self.cache.b_field_tor_vacuum_r

    @cached_property
    def delta_b_field_tor_vacuum_r(self):
        """Variation of (vacuum field times major radius in the toroidal field magnet) from the start of the plasma. [T.m]"""
        return NotImplemented

    @property
    def time(self):
        return self.tokamak.time
