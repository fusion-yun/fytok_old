
# This is file is generated from template
from ..common.IDS import IDS

class LHAntennas(IDS):
    r"""Antenna systems for heating and current drive in the Lower Hybrid (LH) frequencies. In the definitions below, the front (or mouth) of the antenna refers to the plasma facing side of the antenna, while the back refers to the waveguides connected side of the antenna (towards the RF generators).
        
        Note: LHAntennas is an ids
    """
    IDS="lh_antennas"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
