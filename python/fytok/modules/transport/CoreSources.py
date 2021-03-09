
from functools import cached_property, lru_cache

import numpy as np
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.util.logger import logger
from spdm.util.LazyProxy import LazyProxy


class CoreSources(PhysicalGraph):
    """CoreSources
    """
    IDS = "core_sources"

    def __init__(self, cache=None, *args, tokamak=None,   **kwargs):
        super().__init__(*args, **kwargs)
        self._tokamak = tokamak
        if not isinstance(cache, PhysicalGraph) or not isinstance(cache, LazyProxy):
            self._cache = PhysicalGraph(cache)
        else:
            self._cache = cache

    def update(self, *args, **kwargs):
        logger.debug("NOT　IMPLEMENTED!")

    class Profiles1D(PhysicalGraph):
        def __init__(self, cache=None,  *args, parent=None,   **kwargs):
            super().__init__(cache, * args, axis=parent._tokamak.grid.rho_tor_norm, **kwargs)
            self._parent = parent

        @property
        def grid(self):
            return self._parent._tokamak.grid

        class Electrons(PhysicalGraph):
            def __init__(self, cache=None,  *args, parent=None,   **kwargs):
                super().__init__(cache, * args, axis=parent.grid.rho_tor_norm, **kwargs)
                self._parent = parent

            @cached_property
            def particles_decomposed(self):
                return PhysicalGraph(
                    implicit_part=Profile(None, axis=self._parent.grid.rho_tor_norm, description={
                                          "name": "electrons.particles_decomposed.implicit_part"}),
                    explicit_part=Profile(None, axis=self._parent.grid.rho_tor_norm, description={
                                          "name": "electrons.particles_decomposed.explicit_part"})
                )

        @cached_property
        def electrons(self):
            return CoreSources.Profiles1D.Electrons(self._cache.electrons, parent=self)

    @cached_property
    def profiles_1d(self):
        return CoreSources.Profiles1D(self._cache.profiles_1d, parent=self)
