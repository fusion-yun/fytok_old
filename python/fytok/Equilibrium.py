
import collections
import functools

import matplotlib.pyplot as plt
import numpy as np
import scipy
from spdm.data.Entry import open_entry
from spdm.util.AttributeTree import AttributeTree
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles1D, Profiles2D
from spdm.util.sp_export import sp_find_module

from .PFActive import PFActive
from .Wall import Wall
#  psi phi pressure f dpressure_dpsi f_df_dpsi j_parallel q magnetic_shear r_inboard r_outboard rho_tor rho_tor_norm dpsi_drho_tor geometric_axis elongation triangularity_upper triangularity_lower volume rho_volume_norm dvolume_dpsi dvolume_drho_tor area darea_dpsi surface trapped_fraction gm1 gm2 gm3 gm4 gm5 gm6 gm7 gm8 gm9 b_field_max beta_pol mass_density


class Equilibrium(AttributeTree):
    """
        Description of a 2D, axi-symmetric, tokamak equilibrium; result of an equilibrium code.
        imas dd version 3.28
        ids=equilibrium
    """

    @staticmethod
    def __new__(cls,  *args,  backend=None, **kwargs):
        if cls is not Equilibrium:
            return super(Equilibrium, cls).__new__(cls)

        if backend is None:
            backend = "FreeGS"

        plugin_name = f"{__package__}.plugins.equilibrium.Plugin{backend}"

        n_cls = sp_find_module(plugin_name, fragment=f"Equilibrium{backend}")

        if n_cls is None:
            raise ModuleNotFoundError(f"Can not find plugin {plugin_name}#Equilibrium{backend}")

        return AttributeTree.__new__(n_cls)

    def __init__(self,   *args, backend=None, **kwargs):
        super().__init__()

        if len(args)+len(kwargs) > 0:
            self.load(*args, **kwargs)

    def load(self, entry=None,  tokamak=None, nr=129, nz=129, **kwargs):
        self.tokamak = tokamak
        lim_r = self.tokamak.wall.limiter.outline.r
        lim_z = self.tokamak.wall.limiter.outline.z
        self.coordinate_system.grid.dim1 = np.linspace(min(lim_r), max(lim_r), nr)
        self.coordinate_system.grid.dim2 = np.linspace(min(lim_z), max(lim_z), nz)
        return self

    def solve(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def oxpoints(self):
        return NotImplemented

    @property
    def r(self):
        return NotImplemented

    @property
    def z(self):
        return NotImplemented

    @property
    def psi(self):
        return NotImplemented

    @property
    def fvec(self):
        return self.tokamak.vacuum_toroidal_field.r0() * self.tokamak.vacuum_toroidal_field.b0()

    def plot(self, axis=None, *args, levels=40, oxpoints=True, **kwargs):
        """ learn from freegs
        """
        if axis is None:
            axis = plt.gca()

        R = self.r
        Z = self.z
        Psi = self.psi

        levels = np.linspace(np.amin(Psi), np.amax(Psi), levels)

        axis.contour(R, Z, Psi, levels=levels, linewidths=0.2)

        if oxpoints:
            opts, xpts = self.oxpoints

            if len(opts) > 0:
                axis.plot([p[0] for p in opts], [p[1] for p in opts], 'g.', label="O-points")

            if len(xpts) > 0:
                axis.plot([p[0] for p in xpts], [p[1] for p in xpts], 'rx', label="X-points")
                psi_bndry = xpts[0][2]
                axis.contour(R, Z, Psi, levels=[psi_bndry], colors='r', linestyles='dashed', linewidths=0.4)
                axis.plot([], [], 'r--', label="Separatrix")

        return axis
