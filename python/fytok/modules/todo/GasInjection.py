

from  fytok._imas import _T_ids

class GasInjection(IDS):
    r"""Gas injection by a system of pipes and valves
        
        Note: GasInjection is an ids
    """
    _IDS="gas_injection"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
