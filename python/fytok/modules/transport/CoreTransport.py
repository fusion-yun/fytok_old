import collections
from functools import cached_property, lru_cache

import numpy as np
from spdm.data.List import List
from spdm.data.Function import Function
from spdm.data.AttributeTree import AttributeTree
from spdm.util.logger import logger

from ...Profiles import Profiles
from ...RadialGrid import RadialGrid


class CoreTransport(AttributeTree):
    r"""
        Core plasma transport of particles, energy, momentum and poloidal flux. The transport of particles, energy and momentum is described by
        diffusion coefficients,  :math:`D`, and convection velocities,  :math:`v`. These are defined by the total fluxes of particles, energy and momentum, across a
        flux surface given by : :math:`V^{\prime}\left[-DY^{\prime}\left|\nabla\rho_{tor,norm}\right|^{2}+vY\left|\nabla\rho_{tor,norm}\right|\right]`,
        where  :math:`Y` represents the particles, energy and momentum density, respectively, while  :math:`V` is the volume inside a flux surface, the primes denote 
        derivatives with respect to :math:`\rho_{tor,norm}` and
        :math:`\left\langle X\right\rangle` is the flux surface average of a quantity  :math:`X`. This formulation remains valid when changing simultaneously  
        :math:`\rho_{tor,norm}` into :math:`\rho_{tor}`
        in the gradient terms and in the derivatives denoted by the prime. The average flux stored in the IDS as sibling of  :math:`D` and  :math:`v` is the total
        flux described above divided by the flux surface area :math:`V^{\prime}\left\langle \left|\nabla\rho_{tor,norm}\right|\right\rangle` . 
        Note that the energy flux includes the energy transported by the particle flux.

        Attributes :
            profiles_1d
    """
    IDS = "core_transport"

    def __init__(self,  *args, grid: RadialGrid = None,  time=None,   **kwargs):
        super().__init__(*args, **kwargs)
        self._grid = grid
        self._time = time or 0.0

    def update(self, *args, time=None, ** kwargs):
        logger.debug(f"Update {self.__class__.__name__}")
        if time is not None:
            self._time = time

    @cached_property
    def identifier(self):
        return AttributeTree({
            "index": self["identifier.index"] or 0,
            "name": self["identifier.name"] or "",
            "description": self["identifier.description"] or "",
        }, parent=self)

    @property
    def time(self) -> float:
        return self._time

    @cached_property
    def grid_d(self):
        """Grid for effective diffusivities and parallel conductivity"""
        return self._grid.pullback(0.5*(self._grid.psi_norm[:-1]+self._grid.psi_norm[1:]))

    @cached_property
    def grid_v(self):
        """ Grid for effective convections  """
        return self._grid.pullback(self._grid.psi_norm)

    @cached_property
    def grid_flux(self):
        """ Grid for fluxes  """
        return self._grid.pullback(0.5*(self._grid.psi_norm[:-1]+self._grid.psi_norm[1:]))

    class TransportCoeff(Profiles):
        def __init__(self, *args, **kwargs):
            super().__init__(*args,   **kwargs)

        @cached_property
        def d(self):
            return Function(self._parent.grid_d.rho_tor_norm, self["d"])

        @cached_property
        def v(self):
            return Function(self._parent.grid_v.rho_tor_norm, self["v"])

        @cached_property
        def flux(self):
            return Function(self._parent.grid_flux.rho_tor_norm, self["flux"])

    class Ion(Profiles):
        def __init__(self,   *args,   z_ion=1, label=None, neutral_index=None,  **kwargs):
            super().__init__(*args, **kwargs)
            self._label = label or self._data.get("label", None)
            self._z_ion = z_ion or self._data.get("z_ion", None)
            self._neutral_index = neutral_index or self._data.get("neutral_index", None)

        @property
        def z_ion(self):
            """Ion charge (of the dominant ionisation state; lumped ions are allowed),
            volume averaged over plasma radius {dynamic} [Elementary Charge Unit]  FLT_0D  """
            return self._z_ion

        @property
        def label(self):
            """String identifying ion (e.g. H+, D+, T+, He+2, C+, ...) {dynamic}    """
            return self._label

        @property
        def neutral_index(self):
            """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """
            return self._neutral_index

        @cached_property
        def particles(self):
            return CoreTransport.TransportCoeff(self["particles"], parent=self._parent)

        @cached_property
        def energy(self):
            return CoreTransport.TransportCoeff(self["energy"],  parent=self._parent)

    class Electrons(Profiles):
        def __init__(self,   *args,  **kwargs):
            super().__init__(*args, **kwargs)

        @cached_property
        def particles(self):
            return CoreTransport.TransportCoeff(self["particles"], parent=self._parent)

        @cached_property
        def energy(self):
            return CoreTransport.TransportCoeff(self["energy"],  parent=self._parent)

    class Neutral(Profiles):
        def __init__(self,   *args,  **kwargs):
            super().__init__(*args, **kwargs)

    @cached_property
    def electrons(self):
        """ Transport quantities related to the electrons """
        return CoreTransport.Electrons(self['electrons'], parent=self)

    @cached_property
    def ion(self):
        """ Transport coefficients related to the various ion species """
        return List(self['ion'], default_factory=CoreTransport.Ion, parent=self)

    @cached_property
    def neutral(self):
        """ Transport coefficients related to the various neutral species """
        return List(self['neutral'], default_factory=CoreTransport.Neutral,  parent=self)

    @cached_property
    def total_ion_energy(self):
        """ Transport coefficients for the total (summed over ion species) energy equation """
        return CoreTransport.TransportCoeff(self["total_ion_energy"], parent=self)

    @cached_property
    def momentum_tor(self):
        """ Transport coefficients for total toroidal momentum equation  """
        return CoreTransport.TransportCoeff(self["momentum_tor"], parent=self)

    @cached_property
    def conductivity_parallel(self):
        return Function(self.grid_d.rho_tor_norm, self["conductivity_parallel"] or 0.0)

    @cached_property
    def e_field_radial(self):
        """ Radial component of the electric field (calculated e.g. by a neoclassical model) {dynamic} [V.m^-1]"""
        return Function(self.grid_flux.rho_tor_norm, self["e_field_radial"] or 0.0)
