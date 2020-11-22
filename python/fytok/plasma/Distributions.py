
# This is file is generated from template
from fytok.IDS import IDS

class Distributions(IDS):
    r"""Distribution function(s) of one or many particle species. This structure is specifically designed to handle non-Maxwellian distribution function generated during heating and current drive, typically solved using a Fokker-Planck calculation perturbed by a heating scheme (e.g. IC, EC, LH, NBI, or alpha heating) and then relaxed by Coloumb collisions.    8
        .. note:: Distributions is a ids
    """
    IDS="distributions"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
