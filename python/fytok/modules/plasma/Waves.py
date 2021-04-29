
# This is file is generated from template
from ..utilities.IDS import IDS

class Waves(IDS):
    r"""RF wave propagation and deposition. Note that current estimates in this IDS are a priori not taking into account synergies between multiple sources (a convergence loop with Fokker-Planck calculations is required to account for such synergies)
        Note: Waves is an ids
    """
    IDS="waves"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
