
# This is file is generated from template
from ...IDS import IDS

class CoilsNonAxisymmetric(IDS):
    r"""Non axisymmetric active coils system (e.g. ELM control coils, error field correction coils, ...)
        
        Note: CoilsNonAxisymmetric is an ids
    """
    _IDS="coils_non_axisymmetric"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
