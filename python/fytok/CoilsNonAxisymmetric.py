
# This is file is generated from template
from fytok.IDS import IDS

class CoilsNonAxisymmetric(IDS):
    r"""Non axisymmetric active coils system (e.g. ELM control coils, error field correction coils, ...)
        
        .. note:: CoilsNonAxisymmetric is an ids
    """
    IDS="coils_non_axisymmetric"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
