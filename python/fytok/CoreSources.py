
from functools import cached_property, lru_cache

import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles, Profile
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
            super().__init__(cache, * args, axis=parent._tokamak.grid.rho_tor_norm, **kwargs)
            self._parent = parent

        @property
        def grid(self):
            return self._parent._tokamak.grid

        class Electrons(Profiles):
            def __init__(self, cache=None,  *args, parent=None,   **kwargs):
                super().__init__(cache, * args, axis=parent.grid.rho_tor_norm, **kwargs)
                self._parent = parent

            @cached_property
            def particles_decomposed(self):
                return AttributeTree(
                    implicit_part=Profile(self._parent.grid.rho_tor_norm, description={
                                          "name": "electrons.particles_decomposed.implicit_part"}),
                    explicit_part=Profile(self._parent.grid.rho_tor_norm, description={
                                          "name": "electrons.particles_decomposed.explicit_part"})
                )

        @cached_property
        def electrons(self):
            return CoreSources.Profiles1D.Electrons(self._cache.electrons, parent=self)

    @cached_property
    def profiles_1d(self):
        return CoreSources.Profiles1D(self._cache.profiles_1d, parent=self)
