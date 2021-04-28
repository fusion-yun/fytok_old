import collections
from functools import cached_property, lru_cache

import numpy as np
from spdm.data.Node import List
from spdm.data.Function import Function
from spdm.data.AttributeTree import AttributeTree
from spdm.util.logger import logger

from spdm.data.Profiles import Profiles
from ...RadialGrid import RadialGrid
from .ParticleSpecies import Species


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
    """
    IDS = "core_transport"

    def __init__(self,  *args, grid: RadialGrid = None,  **kwargs):
        super().__init__(*args, **kwargs)
        self._grid = grid

    @cached_property
    def identifier(self):
        return self["identifier"]

    @property
    def time(self):
        return np.asarray([profile.time for profile in self.profiles_1d])

    class Profiles1D(AttributeTree):
        def __init__(self, *args, grid=None,  **kwargs):
            super().__init__(*args,   **kwargs)
            self._grid = grid or self._parent._grid
            self._time = self["time"] or 0.0

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

        class Electrons(Profiles):
            def __init__(self,   *args,  **kwargs):
                super().__init__(*args, **kwargs)

            @cached_property
            def particles(self):
                return CoreTransport.Profiles1D.TransportCoeff(self["particles"], parent=self._parent)

            @cached_property
            def energy(self):
                return CoreTransport.Profiles1D.TransportCoeff(self["energy"],  parent=self._parent)

        class Ion(Species):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            @property
            def z_ion(self):
                """Ion charge (of the dominant ionisation state; lumped ions are allowed),
                volume averaged over plasma radius {dynamic} [Elementary Charge Unit]  FLT_0D  """
                return self.__raw_get__("z_ion")

            @property
            def neutral_index(self):
                """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """
                return self.__raw_get__("neutral_index")

            @cached_property
            def particles(self):
                return CoreTransport.Profiles1D.TransportCoeff(self["particles"], parent=self._parent)

            @cached_property
            def energy(self):
                return CoreTransport.Profiles1D.TransportCoeff(self["energy"],  parent=self._parent)

            class Momentum(Profiles):
                def __init__(self, *args,  **kwargs):
                    super().__init__(*args,  **kwargs)

                @cached_property
                def radial(self):
                    return CoreTransport.TransportCoeff(self["radial"], parent=self._parent)

                @cached_property
                def diamagnetic(self):
                    return CoreTransport.TransportCoeff(self["diamagnetic"], parent=self._parent)

                @cached_property
                def parallel(self):
                    return CoreTransport.TransportCoeff(self["parallel"], parent=self._parent)

                @cached_property
                def poloidal(self):
                    return CoreTransport.TransportCoeff(self["poloidal"], parent=self._parent)

                @cached_property
                def toroidal(self):
                    return CoreTransport.TransportCoeff(self["toroidal"], parent=self._parent)

            @cached_property
            def momentum(self):
                return CoreTransport.Profiles1D.Ion.Momentum(self["momentum"],  parent=self._parent)

        class Neutral(Species):
            def __init__(self,   *args,  **kwargs):
                super().__init__(*args, **kwargs)

            @property
            def ion_index(self):
                """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """
                return self.__raw_get__("ion_index")

            @cached_property
            def particles(self):
                return CoreTransport.TransportCoeff(self["particles"], parent=self._parent)

            @cached_property
            def energy(self):
                return CoreTransport.TransportCoeff(self["energy"],  parent=self._parent)

        @cached_property
        def electrons(self):
            """ Transport quantities related to the electrons """
            return CoreTransport.Profiles1D.Electrons(self['electrons'], parent=self)

        @cached_property
        def ion(self):
            """ Transport coefficients related to the various ion species """
            return List(self['ion'], default_factory=CoreTransport.Profiles1D.Ion, parent=self)

        @cached_property
        def neutral(self):
            """ Transport coefficients related to the various neutral species """
            return List(self['neutral'], default_factory=CoreTransport.Profiles1D.Neutral,  parent=self)

        @cached_property
        def total_ion_energy(self):
            """ Transport coefficients for the total (summed over ion species) energy equation """
            return CoreTransport.Profiles1D.TransportCoeff(self["total_ion_energy"], parent=self)

        @cached_property
        def momentum_tor(self):
            """ Transport coefficients for total toroidal momentum equation  """
            return CoreTransport.Profiles1D.TransportCoeff(self["momentum_tor"], parent=self)

        @cached_property
        def conductivity_parallel(self):
            return Function(self.grid_d.rho_tor_norm, self["conductivity_parallel"])

        @cached_property
        def e_field_radial(self):
            """ Radial component of the electric field (calculated e.g. by a neoclassical model) {dynamic} [V.m^-1]"""
            return Function(self.grid_flux.rho_tor_norm, self["e_field_radial"])

    @cached_property
    def profiles_1d(self):
        return List(self["profiles_1d"], default_factory=CoreTransport.Profiles1D, parent=self)
