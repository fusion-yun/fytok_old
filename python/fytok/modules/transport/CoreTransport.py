import collections
from functools import cached_property

import numpy as np
from spdm.data.Function import Function
from spdm.data.Node import Dict, List
from spdm.data.Profiles import Profiles
from spdm.data.TimeSeries import TimeSeries
from spdm.util.logger import logger

from ..utilities.IDS import IDS
from .MagneticCoordSystem import RadialGrid
from .ParticleSpecies import Species


class CoreTransportProfiles1D(Profiles):
    def __init__(self, *args, grid=None,  **kwargs):
        super().__init__(*args,   **kwargs)
        self._grid = grid or self._parent._grid
        self._time = self["time"]

    @property
    def time(self) -> float:
        return self._time

    @cached_property
    def grid_d(self) -> RadialGrid:
        """Grid for effective diffusivities and parallel conductivity"""
        return self._grid.pullback(0.5*(self._grid.psi_norm[:-1]+self._grid.psi_norm[1:]))

    @cached_property
    def grid_v(self) -> RadialGrid:
        """ Grid for effective convections  """
        return self._grid.pullback(self._grid.psi_norm)

    @cached_property
    def grid_flux(self) -> RadialGrid:
        """ Grid for fluxes  """
        return self._grid.pullback(0.5*(self._grid.psi_norm[:-1]+self._grid.psi_norm[1:]))

    class TransportCoeff(Dict):
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

    class Electrons(Dict):
        def __init__(self,   *args,  **kwargs):
            super().__init__(*args, **kwargs)

        @cached_property
        def particles(self):
            return CoreTransportProfiles1D.TransportCoeff(self["particles"], parent=self._parent)

        @cached_property
        def energy(self):
            return CoreTransportProfiles1D.TransportCoeff(self["energy"],  parent=self._parent)

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
            return CoreTransportProfiles1D.TransportCoeff(self["particles"], parent=self._parent)

        @cached_property
        def energy(self):
            return CoreTransportProfiles1D.TransportCoeff(self["energy"],  parent=self._parent)

        class Momentum(Profiles):
            def __init__(self, *args,  **kwargs):
                super().__init__(*args,  **kwargs)

            @cached_property
            def radial(self):
                return CoreTransportProfiles1D.TransportCoeff(self["radial"], parent=self._parent)

            @cached_property
            def diamagnetic(self):
                return CoreTransportProfiles1D.TransportCoeff(self["diamagnetic"], parent=self._parent)

            @cached_property
            def parallel(self):
                return CoreTransportProfiles1D.TransportCoeff(self["parallel"], parent=self._parent)

            @cached_property
            def poloidal(self):
                return CoreTransportProfiles1D.TransportCoeff(self["poloidal"], parent=self._parent)

            @cached_property
            def toroidal(self):
                return CoreTransportProfiles1D.TransportCoeff(self["toroidal"], parent=self._parent)

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
            return CoreTransportProfiles1D.TransportCoeff(self["particles"], parent=self._parent)

        @cached_property
        def energy(self):
            return CoreTransportProfiles1D.TransportCoeff(self["energy"],  parent=self._parent)

    @cached_property
    def electrons(self):
        """ Transport quantities related to the electrons """
        return CoreTransport.Profiles1D.Electrons(self['electrons'], parent=self)

    @cached_property
    def ion(self) -> List:
        """ Transport coefficients related to the various ion species """
        return List[CoreTransportProfiles1D.Ion](self['ion'], parent=self)

    @cached_property
    def neutral(self) -> List:
        """ Transport coefficients related to the various neutral species """
        return List[CoreTransportProfiles1D.Neutral](self['neutral'],   parent=self)

    @cached_property
    def total_ion_energy(self):
        """ Transport coefficients for the total (summed over ion species) energy equation """
        return CoreTransportProfiles1D.TransportCoeff(self["total_ion_energy"], parent=self)

    @cached_property
    def momentum_tor(self):
        """ Transport coefficients for total toroidal momentum equation  """
        return CoreTransportProfiles1D.TransportCoeff(self["momentum_tor"], parent=self)

    @cached_property
    def conductivity_parallel(self):
        return Function(self.grid_d.rho_tor_norm, self["conductivity_parallel"])

    @cached_property
    def e_field_radial(self):
        """ Radial component of the electric field (calculated e.g. by a neoclassical model) {dynamic} [V.m^-1]"""
        return Function(self.grid_flux.rho_tor_norm, self["e_field_radial"])


class CoreTransport(IDS):
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
    _IDS = "core_transport"

    Profiles1D = CoreTransportProfiles1D

    def __init__(self,  *args, grid: RadialGrid = None,  **kwargs):
        super().__init__(*args, **kwargs)
        self._grid = grid

    # @property
    # def time(self):
    #     return np.asarray([profile.time for profile in self.profiles_1d])

    @cached_property
    def profiles_1d(self) -> TimeSeries[CoreTransportProfiles1D]:
        return TimeSeries[CoreTransportProfiles1D](self["profiles_1d"],  time=self.time,   parent=self)
