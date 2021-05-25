from ..common.IDS import IDS

class Gyrokinetics(IDS):
    r"""
        Description of a gyrokinetic simulation (delta-f, flux-tube). All quantities within this IDS are normalised (apart from time),
        thus independent of rhostar, consistently with the local approximation and a spectral representation is assumed in the perpendicular plane 
        (i.e. homogeneous turbulence). All quantities are given in the laboratory frame, except the moments of the perturbed distribution function 
        which are given in the rotating frame.
        Note: Gyrokinetics is an ids
    """
    _IDS="gyrokinetics"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
