import collections
from external.SpDB.python.spdm.util.utilities import try_get
from functools import cached_property

import numpy as np
from spdm.data.Function import Function
from spdm.util.logger import logger


class RadialGrid:
    def __init__(self, axis: np.ndarray = None, label: str = "rho_tor_norm", equilibrium=None) -> None:
        self._equilibrium = equilibrium
        if label == "psi_norm":
            self._psi_norm = axis
        else:
            if not isinstance(axis, np.ndarray) and axis == None:
                axis = np.linspace(0, 1.0, len(self._equilibrium.profiles_1d.psi_norm))
            attr = try_get(self._equilibrium.profiles_1d, label, None)
            if attr is None:
                raise AttributeError(f"Can not find attribute {label}!")
            self._psi_norm = Function(attr, self._equilibrium.profiles_1d.psi_norm)(axis)

        self._vacuum_toroidal_field = self._equilibrium.vacuum_toroidal_field

    @property
    def vacuum_toroidal_field(self):
        return self._vacuum_toroidal_field

    @cached_property
    def psi_magnetic_axis(self):
        """Poloidal flux at the magnetic axis  [Wb]."""
        o, _ = self._equilibrium.critical_points
        return o[0].psi

    @cached_property
    def psi_boundary(self):
        """Poloidal flux at the selected plasma boundary  [Wb]."""
        _, x = self._equilibrium.critical_points
        if len(x) > 0:
            return x[0].psi
        else:
            raise ValueError(f"No x-point")

    @cached_property
    def psi_norm(self):
        return self._psi_norm

    @cached_property
    def psi(self):
        return self._psi_norm * (self.psi_boundary-self.psi_magnetic_axis)+self.psi_magnetic_axis

    @cached_property
    def rho_tor_norm(self):
        return self._equilibrium.profiles_1d.rho_tor_norm(self._psi_norm)

    @cached_property
    def rho_tor(self):
        return self._equilibrium.profiles_1d.rho_tor(self._psi_norm)

    @cached_property
    def rho_pol_norm(self):
        return self._equilibrium.profiles_1d.rho_pol_norm(self._psi_norm)

    @cached_property
    def psi(self):
        return self._equilibrium.profiles_1d.psi(self._psi_norm)

    @cached_property
    def volume(self):
        return self._equilibrium.profiles_1d.volume(self._psi_norm)

    @cached_property
    def area(self):
        return self._equilibrium.profiles_1d.area(self._psi_norm)

    @cached_property
    def surface(self):
        return self._equilibrium.profiles_1d.surface(self._psi_norm)

# class RadialGrid(PhysicalGraph):
#     """Radial grid	"""

#     def __init__(self,  rho_tor_norm=None, equilibrium=None,   **kwargs):
#         self._equilibrium = equilibrium

#         if not isinstance(rho_tor_norm, np.ndarray):
#             rho_tor_norm = 129

#         if isinstance(rho_tor_norm, int):
#             # rho_tor_norm = np.linspace(1.0/rho_tor_norm, 1.0, rho_tor_norm, endpoint=False)
#             rho_tor_norm = np.linspace(0.0, 1.0, rho_tor_norm, endpoint=True)

#         elif isinstance(rho_tor_norm, np.ndarray):
#             pass
#         elif isinstance(rho_tor_norm, collections.abc.Sequence):
#             rho_tor_norm = np.array(rho_tor_norm)
#         else:
#             raise TypeError(f"Illegal axis type! Need 'int' or 'ndarray', not {type(rho_tor_norm)}.")

#         self._rho_tor_norm = rho_tor_norm

#     def reset(self, npoints, primary_coordinates=None):
#         pass

#     @cached_property
#     def rho_tor_norm(self):
#         """	Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm,
#         is the toroidal flux coordinate at the equilibrium boundary (LCFS or 99.x % of the LCFS in case of a
#         fixed boundary equilibium calculation, see time_slice/boundary/b_flux_pol_norm in the equilibrium IDS) {dynamic} [-]"""
#         return Field(self._rho_tor_norm, axis=self._rho_tor_norm, description={"name": "rho_tor_norm"})

#     @cached_property
#     def rho_tor(self):
#         """Toroidal flux coordinate. rho_tor = sqrt(b_flux_tor/(pi*b0)) ~ sqrt(pi*r^2*b0/(pi*b0)) ~ r [m].
#            The toroidal field used in its definition is indicated under vacuum_toroidal_field/b0 {dynamic} [m]"""
#         return self.rho_tor_norm * self.rho_tor_boundary

#     @property
#     def rho_pol_norm(self):
#         """Normalised poloidal flux coordinate = sqrt((psi(rho)-psi(magnetic_axis)) / (psi(LCFS)-psi(magnetic_axis))) {dynamic} [-]"""
#         return np.sqrt(self.psi_norm)

#     @cached_property
#     def psi(self):
#         """Poloidal magnetic flux {dynamic} [Wb]. """
#         return self._equilibrium.profiles_1d.interpolate("rho_tor_norm", "psi")(self.rho_tor_norm)

#     @cached_property
#     def psi_norm(self):
#         """Poloidal magnetic flux {dynamic} [Wb]. """
#         return (self.psi-self.psi_magnetic_axis)/(self.psi_boundary-self.psi_magnetic_axis)

#     @cached_property
#     def volume(self):
#         """Volume enclosed inside the magnetic surface {dynamic} [m^3]"""
#         return None

#     @cached_property
#     def area(self):
#         """Cross-sectional area of the flux surface {dynamic} [m^2]"""
#         return None

#     @cached_property
#     def surface(self):
#         """Surface area of the toroidal flux surface {dynamic} [m^2]"""
#         return None

#     @cached_property
#     def psi_magnetic_axis(self):
#         return self._equilibrium.global_quantities.psi_axis

#     @cached_property
#     def psi_boundary(self):
#         return self._equilibrium.global_quantities.psi_boundary

#     @cached_property
#     def rho_tor_boundary(self):
#         return self._equilibrium.profiles_1d.rho_tor[-1]
