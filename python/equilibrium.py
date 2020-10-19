
import numpy
import scipy


class Equilibrium:

    def __init__(self, *args, nsigma=32, ntheta=32, **kwargs):

        self.nsigma = nsigma
        self.ntheta = ntheta

        # define mesh

        self.sigma_m = numpy.linspace(
            0.5/nsigma, 1 - 0.5/nsigma, nsigma, dtype=float)

        self.sigma_g = numpy.linspace(0, 1, nsigma+1, dtype=float)

        self.theta_m = numpy.linspace(
            0.5/ntheta, 1 - 0.5/ntheta, nsigma, dtype=float)

        self.theta_g = numpy.linspace(0, 1, ntheta+1, dtype=float)

        self.initialize_psi()
        self.define_boundary()
        self.iteration_Loop()

    def boundary(self):
        return []

    def limiter(self):
        return []

    def read_gfile(self, fid):
        pass

    def write_gfile(self, fid):
        pass

    @property
    def psi(self):
        return numpy.ndarray()

    @property
    def equilibrium(self):
        return {}

    def initialize_psi(self):
