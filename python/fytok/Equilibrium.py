
import numpy as np
import scipy
import functools
import collections

import matplotlib.pyplot as plt
from spdm.util.sp_export import sp_find_module

from .Wall import Wall
from .PFCoils import PFCoils


class EqProfiles1D(object):
    """
        Equilibrium profiles (1D radial grid) as a function of the poloidal flux

        @ref: equilibrium.time_slice[itime].profiles_1d
    """

    def __init__(self, equilibrium, npsi=129,  *args, **kwargs):
        self._eq = equilibrium
        self._npsi = npsi
        self._psi = np.linspace(1.0/(self._npsi+1), 1.0, npsi)

    @functools.cached_property
    def psi(self):
        return self._psi

    @functools.cached_property
    def psi_nrom(self):
        return (self._psi-self._psi[0])/(self._psi[-1]-self._psi[0])

    @functools.cached_property
    def phi(self):
        raise NotImplementedError()

    @functools.cached_property
    def pressure(self):
        raise NotImplementedError()

    @functools.cached_property
    def f(self):
        raise NotImplementedError()

    @functools.cached_property
    def dpressure_dpsi(self):
        raise NotImplementedError()

    @functools.cached_property
    def f_df_dpsi(self):
        raise NotImplementedError()

    @functools.cached_property
    def j_parallel(self):
        raise NotImplementedError()

    @functools.cached_property
    def q(self):
        raise NotImplementedError()

    @functools.cached_property
    def magnetic_shear(self):
        raise NotImplementedError()

    @functools.cached_property
    def r_inboard(self):
        raise NotImplementedError()

    @functools.cached_property
    def r_outboard(self):
        raise NotImplementedError()

    @functools.cached_property
    def rho_tor(self):
        raise NotImplementedError()

    @functools.cached_property
    def rho_tor_norm(self):
        raise NotImplementedError()

    @functools.cached_property
    def dpsi_drho_tor(self):
        raise NotImplementedError()

    @functools.cached_property
    def geometric_axis(self):
        raise NotImplementedError()

    @functools.cached_property
    def elongation(self):
        raise NotImplementedError()

    @functools.cached_property
    def triangularity_upper(self):
        raise NotImplementedError()

    @functools.cached_property
    def triangularity_lower(self):
        raise NotImplementedError()

    @functools.cached_property
    def volume(self):
        raise NotImplementedError()

    @functools.cached_property
    def rho_volume_norm(self):
        raise NotImplementedError()

    @functools.cached_property
    def dvolume_dpsi(self):
        raise NotImplementedError()

    @functools.cached_property
    def dvolume_drho_tor(self):
        raise NotImplementedError()

    @functools.cached_property
    def area(self):
        raise NotImplementedError()

    @functools.cached_property
    def darea_dpsi(self):
        raise NotImplementedError()

    @functools.cached_property
    def surface(self):
        raise NotImplementedError()

    @functools.cached_property
    def trapped_fraction(self):
        raise NotImplementedError()

    @functools.cached_property
    def gm1(self):
        raise NotImplementedError()

    @functools.cached_property
    def gm2(self):
        raise NotImplementedError()

    @functools.cached_property
    def gm3(self):
        raise NotImplementedError()

    @functools.cached_property
    def gm4(self):
        raise NotImplementedError()

    @functools.cached_property
    def gm5(self):
        raise NotImplementedError()

    @functools.cached_property
    def gm6(self):
        raise NotImplementedError()

    @functools.cached_property
    def gm7(self):
        raise NotImplementedError()

    @functools.cached_property
    def gm8(self):
        raise NotImplementedError()

    @functools.cached_property
    def gm9(self):
        raise NotImplementedError()

    @functools.cached_property
    def b_field_max(self):
        raise NotImplementedError()

    @functools.cached_property
    def beta_pol(self):
        raise NotImplementedError()

    @functools.cached_property
    def mass_density(self):
        raise NotImplementedError()


class Equilibrium:
    """
        Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.
        imas dd version 3.28
        ids=equilibrium
    """

    @staticmethod
    def __new__(cls,  *args,   backend="FreeGS", **kwargs):
        if cls is not Equilibrium:
            return super(Equilibrium, cls).__new__(cls)

        plugin_name = f"{__package__}.plugins.equilibrium.Plugin{backend}"

        n_cls = sp_find_module(plugin_name, fragment=f"Equilibrium{backend}")

        if n_cls is None:
            raise ModuleNotFoundError(f"Can not find plugin {plugin_name}#Equilibrium{backend}")

        return object.__new__(n_cls)

    def __init__(self, tokamak, *args,  **kwargs):
        super().__init__()
        # self._vacuum_toroidal_field = collections.namedtuple("eq_vacuum_toroidal_field", "r0 b0")(R0, Bt0)
        self._tokamak = tokamak
        self._profiles_1d = EqProfiles1D(self)

    @property
    def tokamak(self):
        return self._tokamak

    def solve(self, profiles=None, **kwargs):
        raise NotImplementedError()

    @property
    def global_quantities(self):
        return []

    @property
    def boundary(self):
        return NotImplemented

    @property
    def boundary_separatrix(self):
        return NotImplemented

    @property
    def constraints(self):
        return NotImplemented

    @property
    def profiles_1d(self):
        return self._profiles_1d

    @property
    def profiles_2d(self):
        return NotImplemented

    @property
    def coordinate_system(self):
        return NotImplemented

    @property
    def convergence(self):
        return NotImplemented

    @property
    def R(self):
        return self._R or np.meshgrid(np.linspace(rmin, rmax, NX), np.linspace(zmin, zmax, NY))

    @property
    def Z(self):
        return self._Z or np.meshgrid(np.linspace(rmin, rmax, NX), np.linspace(zmin, zmax, NY))

    @property
    def psi(self):
        return NotImplemented

    def plot(self, axis=None, **kwargs):

        if axis is None:
            axis = plt.gca()

        # axis.contour(self.R, self.Z, self.psi, **kwargs)
        return axis
