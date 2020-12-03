
from functools import cached_property, lru_cache

import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles, make_x_axis


class RadialGrid:
    """Radial grid	"""

    def __init__(self,  rho_tor_norm=None, equilibrium=None,   **kwargs):
        self._equilibrium = equilibrium
        self._rho_tor_norm = make_x_axis(rho_tor_norm)

    @cached_property
    def rho_tor_norm(self):
        """	Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm,
        is the toroidal flux coordinate at the equilibrium boundary (LCFS or 99.x % of the LCFS in case of a
        fixed boundary equilibium calculation, see time_slice/boundary/b_flux_pol_norm in the equilibrium IDS) {dynamic} [-]"""
        return self._rho_tor_norm

    @cached_property
    def rho_tor(self):
        """Toroidal flux coordinate. rho_tor = sqrt(b_flux_tor/(pi*b0)) ~ sqrt(pi*r^2*b0/(pi*b0)) ~ r [m].
           The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0 {dynamic} [m]"""
        return self.rho_tor_norm * self.rho_tor_boundary

    @property
    def rho_pol_norm(self):
        """Normalised poloidal flux coordinate = sqrt((psi(rho)-psi(magnetic_axis)) / (psi(LCFS)-psi(magnetic_axis))) {dynamic} [-]"""
        return np.sqrt(self.psi_norm)

    @cached_property
    def psi(self):
        """Poloidal magnetic flux {dynamic} [Wb]. """
        return self._equilibrium.profiles_1d.interpolate("rho_tor_norm", "psi")(self.rho_tor_norm)

    @cached_property
    def psi_norm(self):
        """Poloidal magnetic flux {dynamic} [Wb]. """
        return (self.psi-self.psi_magnetic_axis)/(self.psi_boundary-self.psi_magnetic_axis)

    @cached_property
    def volume(self):
        """Volume enclosed inside the magnetic surface {dynamic} [m^3]"""
        return None

    @cached_property
    def area(self):
        """Cross-sectional area of the flux surface {dynamic} [m^2]"""
        return None

    @cached_property
    def surface(self):
        """Surface area of the toroidal flux surface {dynamic} [m^2]"""
        return None

    @cached_property
    def psi_magnetic_axis(self):
        return self._equilibrium.global_quantities.psi_axis

    @cached_property
    def psi_boundary(self):
        return self._equilibrium.global_quantities.psi_boundary

    @cached_property
    def rho_tor_boundary(self):
        return self._equilibrium.profiles_1d.rho_tor[-1]
