
from functools import cached_property, lru_cache

import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles


class CoreSources(AttributeTree):
    """CoreSources
    """
    IDS = "core_sources"

    def __init__(self, cache, *args, tokamak=None, **kwargs):
        super().__init__(*args, **kwargs)

    class Profiles1D(Profiles):
        def __init__(self, cache=None,  *args, equilibrium=None, rho_tor_norm=None, **kwargs):

            super().__init__(cache, * args, x_axis=rho_tor_norm, **kwargs)
            self.__dict__["_equilibrium"] = equilibrium
            self.rho_tor_norm = self._x_axis

        class Grid(Profiles):
            """Normalised toroidal flux coordinate. The normalizing value for rho_tor_norm,
            is the toroidal flux coordinate at the equilibrium boundary (LCFS or 99.x % of the LCFS in case of
            a fixed boundary equilibium calculation, see time_slice/boundary/b_flux_pol_norm in the equilibrium IDS) {dynamic} [-]	"""

            def __init__(self, cache=None,  *args, equilibrium=None, npoint=129, **kwargs):
                super().__init__(cache, *args, x_axis=npoint, **kwargs)
                self.__dict__['_equilibrium'] = equilibrium
                self["rho_tor_norm"] = self._x_axis

            def __missing__(self, key):
                res = super().__missing__(key)
                if isinstance(res, np.ndarray):
                    pass
                elif not res:
                    try:
                        res = self._equilibrium.profiles_1d.mapping("rho_tor_norm", key)(self.rho_tor_norm)
                    except LookupError:
                        res = None

                return res

    # @cached_property
    # def profiles_1d(self):
    #     return Profiles1D(self._cache.profiles_1d, parent=self)
