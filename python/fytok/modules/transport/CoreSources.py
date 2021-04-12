
from functools import cached_property, lru_cache
from fytok.modules.utilities.RadialGrid import RadialGrid
import collections
import numpy as np
from spdm.numerical.Function import Function
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.util.logger import logger


class CoreSources(PhysicalGraph):
    """CoreSources
    """
    IDS = "core_sources"

    def __init__(self, *args, grid: RadialGrid = None, time=None,   **kwargs):
        super().__init__(*args, **kwargs)
        self._time = time or 0.0
        self._grid = grid

    def update(self, *args, time=None, ** kwargs):
        logger.debug(f"Update {self.__class__.__name__} [time={time}] at: Do nothing")
        if len(args) > 0 and isinstance(args[0], collections.abc.Mapping):
            self |= args[0]
        if time is not None:
            self._time = time

    @property
    def time(self) -> float:
        return self._time

    @property
    def grid(self) -> RadialGrid:
        return self._grid

    class Particle(PhysicalGraph):
        def __init__(self, *args,  **kwargs):
            super().__init__(*args, **kwargs)

        @cached_property
        def particles(self):
            return Function(self._parent.grid.rho_tor_norm, self["particles"], parent=self._parent)

        @cached_property
        def energy(self):
            return Function(self._parent.grid.rho_tor_norm, self["energy"], parent=self._parent)

        @cached_property
        def momentum(self):
            return PhysicalGraph({
                "radial": Function(self._parent.grid.rho_tor_norm, self["momentum.radial"], parent=self._parent),
                "diamagnetic": Function(self._parent.grid.rho_tor_norm, self["momentum.diamagnetic"], parent=self._parent),
                "parallel": Function(self._parent.grid.rho_tor_norm, self["momentum.parallel"], parent=self._parent),
                "poloidal": Function(self._parent.grid.rho_tor_norm, self["momentum.poloidal"], parent=self._parent),
                "toroidal": Function(self._parent.grid.rho_tor_norm, self["momentum.toroidal"], parent=self._parent)
            })

    class Electrons(Particle):
        def __init__(self,  *args,   **kwargs):
            super().__init__(* args,  **kwargs)

    class Ion(Particle):
        def __init__(self,  *args,   **kwargs):
            super().__init__(* args,  **kwargs)

    class Neutral(Particle):
        def __init__(self,  *args,   **kwargs):
            super().__init__(* args,  **kwargs)

    @cached_property
    def electrons(self):
        return CoreSources.Electrons(self["electrons"], parent=self)

    @cached_property
    def ion(self):
        return PhysicalGraph([CoreSources.Ion(d, parent=self) for d in self["ion"]], parent=self)

    @cached_property
    def neutral(self):
        return PhysicalGraph([CoreSources.Neutral(d, parent=self) for d in self["neutral"]], parent=self)

    @cached_property
    def total_ion_energy(self):
        res = Function(self.grid.rho_tor_norm,  0.0)
        for ion in self.ion:
            res += ion.energy
        return res

    @cached_property
    def total_ion_power_inside(self):
        return NotImplemented

    @cached_property
    def torque_tor_inside(self):
        return NotImplemented

    @cached_property
    def j_parallel(self):
        return Function(self.grid.rho_tor_norm, self["j_parallel"])

    @cached_property
    def current_parallel_inside(self):
        return Function(self.grid.rho_tor_norm, self["current_parallel_inside"])

    @cached_property
    def conductivity_parallel(self):
        return Function(self.grid.rho_tor_norm, self["conductivity_parallel"])
