import collections
from functools import cached_property

import numpy as np
from spdm.data.Function import Function
from spdm.data.Node import Dict
from spdm.util.logger import logger
from spdm.util.utilities import try_get

from .Misc import VacuumToroidalField


class RadialGrid:
    r"""

    """

    def __init__(self, psi_norm, *args, vacuum_toroidal_field: VacuumToroidalField = None, equilibrium=None, **kwargs) -> None:
        self._data = kwargs
        if equilibrium is not None:

            self._vacuum_toroidal_field = vacuum_toroidal_field
            self._equilibrium = equilibrium
            self._psi_axis = equilibrium.boundary.psi_axis
            self._psi_boundary = equilibrium.boundary.psi_boundary
            if psi_norm is None:
                psi_norm = equilibrium.psi_norm
        else:
            self._equilibrium = None
            self._vacuum_toroidal_field = vacuum_toroidal_field or VacuumToroidalField(1.0, 1.0)
            self._psi_axis = kwargs.get("psi_axis", NotImplemented)
            self._psi_boundary = kwargs.get("psi_boundary", NotImplemented)
        assert(self._equilibrium != None)
        if isinstance(psi_norm, int):
            self._psi_norm = np.linspace(0, 1.0, psi_norm)
        elif isinstance(psi_norm, np.ndarray):
            self._psi_norm = psi_norm
        else:
            self._psi_norm = np.linspace(0, 1.0, 128)
            # raise ValueError(f"psi_norm is not defined")

    def _try_get(self, k):
        d = self._data.get(k, None)
        if d is None:
            d = try_get(self._equilibrium, k, None)

        if isinstance(d, Function):
            if d.x is not self._psi_norm:
                d = d(self._psi_norm)
        elif d is None or (not isinstance(d, np.ndarray) and d == None):
            raise AttributeError(f"Can not find {k} in {type(self._equilibrium)}!")
        elif not isinstance(d, np.ndarray):
            raise TypeError(type(d))
        elif d.shape != self._psi_norm.shape:
            raise RuntimeError(f"Illegal shape! {k} {d.shape }!={self._psi_norm.shape}")

        return d.view(np.ndarray)

    def pullback(self, psi_norm):
        return RadialGrid(psi_norm, equilibrium=self._equilibrium, **self._data)

    @property
    def vacuum_toroidal_field(self):
        return self._vacuum_toroidal_field

    @cached_property
    def psi_magnetic_axis(self):
        """Poloidal flux at the magnetic axis  [Wb]."""
        return self._psi_axis

    @cached_property
    def psi_boundary(self):
        """Poloidal flux at the selected plasma boundary  [Wb]."""
        return self._psi_boundary

    @property
    def psi_norm(self):
        return self._psi_norm

    @cached_property
    def psi(self):
        """Poloidal magnetic flux {dynamic} [Wb]. This quantity is COCOS-dependent, with the following transformation"""
        return self.psi_norm * (self.psi_boundary-self.psi_magnetic_axis)+self.psi_magnetic_axis

    @cached_property
    def rho_tor_norm(self):
        r"""Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm, is the toroidal flux coordinate 
            at the equilibrium boundary (LCFS or 99.x % of the LCFS in case of a fixed boundary equilibium calculation, 
            see time_slice/boundary/b_flux_pol_norm in the equilibrium IDS) {dynamic} [-]
        """
        return self._try_get("rho_tor_norm")

    @cached_property
    def rho_tor(self):
        r"""Toroidal flux coordinate. rho_tor = sqrt(b_flux_tor/(pi*b0)) ~ sqrt(pi*r^2*b0/(pi*b0)) ~ r [m]. 
            The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0 {dynamic} [m]"""
        return self._try_get("rho_tor")

    @cached_property
    def rho_pol_norm(self):
        r"""Normalised poloidal flux coordinate = sqrt((psi(rho)-psi(magnetic_axis)) / (psi(LCFS)-psi(magnetic_axis))) {dynamic} [-]"""
        return self._try_get("rho_pol_norm")

    @cached_property
    def area(self):
        """Cross-sectional area of the flux surface {dynamic} [m^2]"""
        return self._try_get("area")

    @cached_property
    def surface(self):
        """Surface area of the toroidal flux surface {dynamic} [m^2]"""
        return self._try_get("surface")

    @cached_property
    def volume(self):
        """Volume enclosed inside the magnetic surface {dynamic} [m^3]"""
        return self._try_get("volume")
