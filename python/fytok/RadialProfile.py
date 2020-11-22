
import collections
import copy
from functools import cached_property
import numpy as np
import scipy
from scipy import constants
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from spdm.util.AttributeTree import AttributeTree, _last_, _next_
from spdm.util.Interpolate import Interpolate1D, derivate, integral
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module

from fytok.Identifier import Identifier


class RadialProfile(AttributeTree):
    def __init__(self, cache, *args, primary_coordinate=None, parent=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__["_parent"] = parent

        if primary_coordinate is None:
            primary_coordinate = Identifier(name="psi_norm", index=-1, description="")
        elif isinstance(primary_coordinate, str):
            primary_coordinate = Identifier(name=primary_coordinate, index=-1, description="")
        elif isinstance(primary_coordinate, Identifier):
            pass
        elif isinstance(primary_coordinate, collections.abc.Mapping):
            primary_coordinate = Identifier(primary_coordinate)
        else:
            raise TypeError(f"{type(primary_coordinate)}")
        self.primary_coordinate = primary_coordinate

    class Grid(AttributeTree):
        def __init__(self, cache=None,  *args, equilibrium=None, primary_coordinate=None,  **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__['_eq'] = equilibrium
            self.__dict__['_primary_coordinate'] = primary_coordinate

            """Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm,
                    is the toroidal flux coordinate at the equilibrium boundary (LCFS or 99.x % of the LCFS in case of
                    a fixed boundary equilibium calculation, see time_slice/boundary/b_flux_pol_norm in the equilibrium IDS) {dynamic} [-]	"""
            if rho_tor_norm is None:
                rho_tor_norm = 129
            if type(rho_tor_norm) is int:
                self.rho_tor_norm = np.linspace(0, 1.0, rho_tor_norm)
            else:
                self.rho_tor_norm = rho_tor_norm

            self |= {
                "psi_magnetic_axis": self._eq.global_quantities.psi_axis,  # Value of the poloidal magnetic flux at the magnetic axis (useful to normalize the psi array values when the radial grid doesn't go from the magnetic axis to the plasma boundary) {dynamic} [Wb]	FLT_0D"""
                "psi_boundary": self._eq.global_quantities.psi_boundary    # Value of the poloidal magnetic flux at the plasma boundary (useful to normalize the psi array values when the radial grid doesn't go from the magnetic axis to the plasma boundary) {dynamic} [Wb]	FLT_0D"""
            }

        def __missing__(self, key):
            path = key.split('.')
            x = self._eq.profiles_1d.rho_tor_norm
            y = self._eq.profiles_1d[key]
            if y is None or y is NotImplemented:
                raise KeyError(path)
            elif not isinstance(y, np.ndarray) or len(y.shape) > 1:
                return y
            else:
                return UnivariateSpline(x, y)(self.rho_tor_norm)

        # def rho_tor(self):
        #     """Toroidal flux coordinate. rho_tor = sqrt(b_flux_tor/(pi*b0)) ~ sqrt(pi*r^2*b0/(pi*b0)) ~ r [m].
        #     The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0 {dynamic} [m]"""

        # def rho_pol_norm(self):
        #     """Normalised poloidal flux coordinate = sqrt((psi(rho)-psi(magnetic_axis)) / (psi(LCFS)-psi(magnetic_axis))) {dynamic} [-]"""
        #     return NotImplemented

        # def psi(self):
        #     """Poloidal magnetic flux {dynamic} [Wb]. """

        # def volume(self):
        #     """Volume enclosed inside the magnetic surface {dynamic} [m^3]"""

        # def area(self):
        #     """Cross-sectional area of the flux surface {dynamic} [m^2]"""

        # def surface(self):
        #     """Surface area of the toroidal flux surface {dynamic} [m^2]"""

    @cached_property
    def grid(self):
        return RadialProfiles.Grid(self)
