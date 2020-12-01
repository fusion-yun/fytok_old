
from functools import cached_property, lru_cache

import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles
from .RadialGrid import RadialGrid


class CoreTransport(AttributeTree):
    r"""Core Transport

    Todo:
        * transport
        * need complete

    """
    IDS = "core_transport"

    def __init__(self, config=None, *args,   rho_tor_norm=None, tokamak=None,  **kwargs):
        super().__init__(*args, **kwargs)
        self._tokamak = tokamak
        self._rho_tor_norm = rho_tor_norm
        self._psi_axis = self._tokamak.global_quantities.psi_axis
        self._psi_boundary = self._tokamak.global_quantities.psi_boundary

    class TransportCoeff(AttributeTree):
        def __init__(self, cache=None, *args, parent=None, **kwargs):
            self._parent = parent

        @cached_property
        def d(self):
            return np.zeros(self._parent.grid_d.rho_tor_norm.shape)

        @cached_property
        def v(self):
            return np.zeros(self._parent.grid_v.rho_tor_norm.shape)

        @cached_property
        def flux(self):
            return np.zeros(self._parent.grid_flux.rho_tor_norm.shape)

    class Particle(AttributeTree):
        def __init__(self, cache, *args, parent=None, **kwargs):
            self._parent = parent

        @cached_property
        def particles(self):
            return CoreTransport.TransportCoeff(self._cache.pareticles, parent=self._parent)

        @cached_property
        def energy(self):
            return CoreTransport.TransportCoeff(self._cache.energy, parent=self._parent)

    class Ion(Particle):
        def __init__(self,   *args,  **kwargs):
            super().__init__(*args, **kwargs)

    class Electrons(Particle):
        def __init__(self,   *args,  **kwargs):
            super().__init__(*args, **kwargs)

    class Neutral(Particle):
        def __init__(self,   *args,  **kwargs):
            super().__init__(*args, **kwargs)

    class Profiles1D(Profiles):
        def __init__(self, cache=None, *args,
                     rho_tor_norm=None,
                     psi_axis=None,
                     psi_boundary=None, **kwargs):
            super().__init__(cache, * args, x_axis=rho_tor_norm, **kwargs)
            self.__dict__["_psi_axis"] = psi_axis
            self.__dict__["_psi_boundary"] = psi_boundary

        @cached_property
        def grid_d(self):
            return RadialGrid(self._cache.grid_d,
                             rho_tor_norm=self._x_axis,
                             psi_axis=self._psi_axis,
                             psi_boundary=self._psi_boundary)

        @cached_property
        def grid_v(self):
            return RadialGrid(self._cache.grid_v, rho_tor_norm=self._x_axis, psi_axis=self._psi_axis, psi_boundary=self._psi_boundary)

        @cached_property
        def grid_flux(self):
            return RadialGrid(self._cache.grid_flux, rho_tor_norm=self._x_axis, psi_axis=self._psi_axis, psi_boundary=self._psi_boundary)

        @cached_property
        def ion(self):
            return AttributeTree(default_factory_array=lambda _holder=self: CoreTransport.Ion(parent=_holder))

        @cached_property
        def electrons(self):
            return AttributeTree(default_factory_array=lambda _holder=self: CoreTransport.Electrons(parent=_holder))

        @cached_property
        def neutral(self):
            return AttributeTree(default_factory_array=lambda _holder=self: CoreTransport.Neutral(parent=_holder))

        @cached_property
        def total_ion_energy(self):
            return CoreTransport.TransportCoeff(self._cache.total_ion_energy, parent=self._parent)

        @cached_property
        def momentum_tor(self):
            return CoreTransport.TransportCoeff(self._cache.momentum_tor, parent=self._parent)

    @cached_property
    def profiles_1d(self):
        return CoreTransport.Porfiles1D(self._cache.profiles_1d,
                                        parent=self,
                                        rho_tor_norm=self._rho_tor_norm,
                                        psi_axis=self._psi_axis,
                                        psi_boundary=self._psi_boundary
                                        )
