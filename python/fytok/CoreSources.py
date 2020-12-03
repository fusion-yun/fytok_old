
from functools import cached_property, lru_cache

import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles
from spdm.util.LazyProxy import LazyProxy


class CoreSources(AttributeTree):
    """CoreSources
    """
    IDS = "core_sources"

    def __init__(self, cache=None, *args, tokamak=None,   **kwargs):
        super().__init__(*args, **kwargs)
        self._tokamak = tokamak
        if not isinstance(cache, AttributeTree) or not isinstance(cache, LazyProxy):
            self._cache = AttributeTree(cache)
        else:
            self._cache = cache

    def update(self, *args, **kwargs):
        logger.debug("NOT　IMPLEMENTED!")

    class Profiles1D(Profiles):
        def __init__(self, cache=None,  *args, parent=None,   **kwargs):
            super().__init__(cache, * args, x_axis=parent._tokamak.grid.rho_tor_norm, **kwargs)
            self._parent = parent

        @property
        def grid(self):
            return self._parent._tokamak.grid

    @cached_property
    def profiles_1d(self):
        return CoreSources.Profiles1D(self._cache.profiles_1d, parent=self)
