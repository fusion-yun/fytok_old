
from functools import cached_property, lru_cache

import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles


class RadialGrid(Profiles):
    """Radial grid	"""

    def __init__(self, cache=None,  *args, psi_axis=None, psi_boundary=None, x_axis=129, **kwargs):
        super().__init__(cache, *args, x_axis=x_axis, **kwargs)
        self._psi_axis = psi_axis
        self._psi_boundary = psi_boundary

    @property
    def psi_axis(self):
        return self._psi_axis

    @property
    def psi_magnetic_axis(self):
        return self._psi_axis

    @property
    def psi_boundary(self):
        return self._psi_boundary

    # @property
    # def rho_tor(self):
    #     """Toroidal flux coordinate. rho_tor = sqrt(b_flux_tor/(pi*b0)) ~ sqrt(pi*r^2*b0/(pi*b0)) ~ r [m].
    #     The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0 {dynamic} [m]"""
    #     return None

    @property
    def rho_tor_norm(self):
        """	Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm, 
         is the toroidal flux coordinate at the equilibrium boundary (LCFS or 99.x % of the LCFS in case of a 
          fixed boundary equilibium calculation, see time_slice/boundary/b_flux_pol_norm in the equilibrium IDS) {dynamic} [-]"""

        return self._x_axis

    # @property
    # def rho_pol_norm(self):
    #     """Normalised poloidal flux coordinate = sqrt((psi(rho)-psi(magnetic_axis)) / (psi(LCFS)-psi(magnetic_axis))) {dynamic} [-]"""
    #     return None

    # @property
    # def psi(self):
    #     """Poloidal magnetic flux {dynamic} [Wb]. """
    #     return self.cache("psi")

    @property
    def psi_norm(self):
        """Poloidal magnetic flux {dynamic} [Wb]. """
        return (self.psi-self.psi_axis)/(self.psi_boundary-self.psi_axis)

    @property
    def volume(self):
        """Volume enclosed inside the magnetic surface {dynamic} [m^3]"""
        return None

    @property
    def area(self):
        """Cross-sectional area of the flux surface {dynamic} [m^2]"""
        return None

    @property
    def surface(self):
        """Surface area of the toroidal flux surface {dynamic} [m^2]"""
        return None
