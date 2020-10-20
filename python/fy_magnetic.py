import scipy
import collections
from math import sqrt, pi


class FyMagnetic:

    COIL = collections.namedtuple("COIL", "r z current")

    def __init__(self, coils):
        pass

    def psi(self, r, z, coils):
        def green_function(self, Rc, Zc, R, Z):
            k2 = 4*Rc*R/((Rc+R)**2+(Zc-Z)**2)
            k = sqrt(k2)
            return sqrt(R*Rc)/(2.0*pi*k)*((2-k2) * scipy.special.ellipk(k)-2*scipy.special.ellipe(k))

        return sum([green_function(coil.r, coil.z, r, z)*coil.current for coil in coils])
