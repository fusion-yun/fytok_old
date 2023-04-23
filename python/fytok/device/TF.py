

from spdm.data.Dict import Dict
from spdm.data.List import List
from spdm.data.sp_property import sp_property

from ..common.IDS import IDS
from ..common.TimeSeries import TimeSeries


class TFCoil(Dict):
    pass


class TF(IDS):
    """TF　Coils

    """
    _IDS = "tf"

    def __init__(self,  *args,    **kwargs):
        super().__init__(*args, **kwargs)

    r0: float = sp_property()
    """Reference major radius of the device (from the official description of the device).

           This node is the placeholder for this official machine description quantity
            (typically the middle of the vessel at the equatorial midplane, although the exact
            definition may depend on the device) {static} [m]"""

    is_periodic: bool = sp_property()
    """Flag indicating whether coils are described one by one in the coil() structure (flag=0) or whether the coil
        structure represents only coils having different characteristics (flag = 1, n_coils must be filled in that case).
        In the latter case, the coil() sequence is repeated periodically around the torus. {static}	INT_0D	"""

    coil: List[TFCoil] = sp_property()
    """Set of coils around the tokamak {static}	struct_array [max_size=32]	1- 1...N"""

    field_map: Dict = sp_property()
    """Map of the vacuum field at various time slices, represented using the generic grid description {dynamic}
                struct_array [max_size=unbounded]	 """

    b_field_tor_vacuum_r: TimeSeries = sp_property()
    """Vacuum field times major radius in the toroidal field magnet. Positive sign means anti-clockwise when viewed 
            from above [T.m].  """

    delta_b_field_tor_vacuum_r: TimeSeries = sp_property()
    """Variation of (vacuum field times major radius in the toroidal field magnet) from the start of the plasma. [T.m]"""
