
from functools import cached_property, lru_cache

import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.Profiles import Profiles, Profile
from spdm.util.LazyProxy import LazyProxy

from fytok.RadialGrid import RadialGrid


class CoreTransport(AttributeTree):
    r"""Core plasma transport of particles, energy, momentum and poloidal flux. The transport of particles, energy and momentum is described by
        diffusion coefficients,  :math:`D`, and convection velocities,  :math:`v`. These are defined by the total fluxes of particles, energy and momentum, across a
        flux surface given by : :math:`V^{\prime}\left[-DY^{\prime}\left|\nabla\rho_{tor,norm}\right|^{2}+vY\left|\nabla\rho_{tor,norm}\right|\right]`,
        where  :math:`Y` represents the particles, energy and momentum density, respectively, while  :math:`V` is the volume inside a flux surface, the primes denote derivatives with respect to :math:`\rho_{tor,norm}` and
        :math:`\left\langle X\right\rangle` is the flux surface average of a quantity  :math:`X`. This formulation remains valid when changing simultaneously  :math:`\rho_{tor,norm}` into :math:`\rho_{tor}`
        in the gradient terms and in the derivatives denoted by the prime. The average flux stored in the IDS as sibling of  :math:`D` and  :math:`v` is the total
        flux described above divided by the flux surface area :math:`V^{\prime}\left\langle \left|\nabla\rho_{tor,norm}\right|\right\rangle` . Note that the energy flux includes the energy transported by the particle flux.

        Attributes :
            profiles_1d
    """
    IDS = "core_transport"

    def __init__(self, cache=None, *args,   grid=None, tokamak=None,  **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = cache or AttributeTree()
        self._tokamak = tokamak
        self._grid = grid or tokamak.grid()

    def update(self, *args, **kwargs):
        logger.debug("NOT　IMPLEMENTED!")

    class TransportCoeff(Profiles):
        def __init__(self, cache=None, *args, parent=None, **kwargs):

            super().__init__(cache, *args, x_axis=parent.grid_d.rho_tor_norm, **kwargs)
            self._parent = parent
            self.d = Profile(self._parent.grid_d.rho_tor_norm, self._cache.d, description={"name": "d"})
            self.v = Profile(self._parent.grid_v.rho_tor_norm, self._cache.v, description={"name": "v"})
            self.flux = Profile(self._parent.grid_flux.rho_tor_norm, self._cache.flux, description={"name": "flux"})

    class Particle(AttributeTree):
        def __init__(self, cache=None, *args, parent=None, **kwargs):
            self._parent = parent
            self._cache = cache or AttributeTree()

        @cached_property
        def particles(self):
            return CoreTransport.TransportCoeff(self._cache.particles, parent=self._parent)

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
        def __init__(self, cache=None, *args, equilibrium=None, grid=None, **kwargs):
            super().__init__(cache, * args,  **kwargs)
            self.__dict__["_equilibrium"] = equilibrium
            if isinstance(grid, RadialGrid):
                self.__dict__["_grid_d"] = grid
            else:
                self.__dict__["_grid_d"] = self._equilibrium.radial_grid()

        @property
        def grid_d(self):
            """Grid for effective diffusivities and parallel conductivity"""
            return self._grid_d

        @property
        def grid_v(self):
            """ Grid for effective convections

                Todo :
                    FIXME 

            """
            return self._grid_d

        @property
        def grid_flux(self):
            """ Grid for fluxes

                Todo :
                    FIXME 
            """
            return self._grid_d

        @cached_property
        def ion(self):
            """ Ion　: Transport coefficients related to the various ion species """
            return AttributeTree(default_factory_array=lambda _holder=self: CoreTransport.Ion(parent=_holder))

        @cached_property
        def electrons(self):
            """ Electrons :　Transport quantities related to the electrons"""
            return CoreTransport.Electrons(self._cache.electrons, parent=self)

        @cached_property
        def neutral(self):
            """ Neutral : Transport coefficients related to the various neutral species"""
            return AttributeTree(default_factory_array=lambda _holder=self: CoreTransport.Neutral(parent=_holder))

        @cached_property
        def total_ion_energy(self):
            """
                CoreTransport.TransportCoeff : Transport coefficients for the total (summed over ion species) energy equation
            """
            return CoreTransport.TransportCoeff(self._cache.total_ion_energy, parent=self._parent)

        @cached_property
        def momentum_tor(self):
            """ 
                CoreTransport.TransportCoeff : Transport coefficients for total toroidal momentum equation
            """
            return CoreTransport.TransportCoeff(self._cache.momentum_tor, parent=self._parent)

        @property
        def conductivity_parallel(self):
            if "_conductivity_parallel" not in self.__dict__ or self.__dict__["_conductivity_parallel"] is None:
                self.__dict__["_conductivity_parallel"] = Profile(
                    self.grid_d.rho_tor_norm, None,  description={"name": "conductivity_parallel"})
            return self._conductivity_parallel

        @conductivity_parallel.setter
        def conductivity_parallel(self, v):
            self._conductivity_parallel.assign(v)

        @cached_property
        def e_field_radial(self):
            """ Radial component of the electric field (calculated e.g. by a neoclassical model) {dynamic} [V.m^-1]"""
            return Profile(super().cache("e_field_radial"), self.grid_d.rho_tor_norm)

    @cached_property
    def profiles_1d(self):
        """ CoreTransport.Profiles1D : Transport coefficient profiles for various time slices. 
        Fluxes and convection are positive (resp. negative) when outwards i.e. towards the LCFS 
        (resp. inwards i.e. towards the magnetic axes). {dynamic} """

        return CoreTransport.Profiles1D(self._cache.profiles_1d, parent=self, grid=self._grid, equilibrium=self._tokamak.equilibrium)
