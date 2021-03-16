
from functools import cached_property, lru_cache

import numpy as np
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.data.Field import Field

from spdm.util.logger import logger


class CoreSources(PhysicalGraph):
    """CoreSources
    """
    IDS = "core_sources"

    def __init__(self,  *args,   **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, *args, **kwargs):
        logger.debug("NOT　IMPLEMENTED!")

    class Profiles1D(PhysicalGraph):
        def __init__(self,   *args,   **kwargs):
            super().__init__(* args, **kwargs)

        @property
        def grid(self):
            return self._parent._tokamak.grid

        class Electrons(PhysicalGraph):
            def __init__(self,  *args,   **kwargs):
                super().__init__(* args,  **kwargs)

            @cached_property
            def particles_decomposed(self):
                return PhysicalGraph(
                    implicit_part=Field(None, axis=self._parent.grid.rho_tor_norm, description={
                                          "name": "electrons.particles_decomposed.implicit_part"}),
                    explicit_part=Field(None, axis=self._parent.grid.rho_tor_norm, description={
                                          "name": "electrons.particles_decomposed.explicit_part"})
                )

        @cached_property
        def electrons(self):
            return CoreSources.Profiles1D.Electrons(self["electrons"], parent=self)

    @cached_property
    def profiles_1d(self):
        return CoreSources.Profiles1D(self["profiles_1d"], parent=self)
