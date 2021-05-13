import collections
from functools import cached_property

import numpy as np
from fytok.modules.utilities.Combiner import Combiner
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Function import Function
from spdm.data.Node import Dict, List
from spdm.data.Profiles import Profiles
from spdm.data.TimeSeries import TimeSeries, TimeSlice
from spdm.util.logger import logger

from ..utilities.IDS import IDS, IDSCode
from ..utilities.Misc import Identifier
from .MagneticCoordSystem import RadialGrid
from .ParticleSpecies import Species, SpeciesIon


class TransportCoeff(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,   **kwargs)

    @cached_property
    def d(self) -> Function:
        return Function(self._parent.grid_d.rho_tor_norm, self["d"])

    @cached_property
    def v(self) -> Function:
        return Function(self._parent.grid_v.rho_tor_norm, self["v"])

    @cached_property
    def flux(self) -> Function:
        return Function(self._parent.grid_flux.rho_tor_norm, self["flux"])


class CoreTransportElectrons(Dict):
    def __init__(self,   *args,  **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def particles(self) -> TransportCoeff:
        return TransportCoeff(self["particles"], parent=self._parent)

    @cached_property
    def energy(self) -> TransportCoeff:
        return TransportCoeff(self["energy"],  parent=self._parent)


class CoreTransportIon(SpeciesIon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def particles(self) -> TransportCoeff:
        return TransportCoeff(self["particles"], parent=self._parent)

    @cached_property
    def energy(self) -> TransportCoeff:
        return TransportCoeff(self["energy"],  parent=self._parent)


class CoreTransportMomentum(Profiles):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args,  **kwargs)

    @cached_property
    def radial(self) -> TransportCoeff:
        return TransportCoeff(self["radial"], parent=self._parent)

    @cached_property
    def diamagnetic(self) -> TransportCoeff:
        return TransportCoeff(self["diamagnetic"], parent=self._parent)

    @cached_property
    def parallel(self) -> TransportCoeff:
        return TransportCoeff(self["parallel"], parent=self._parent)

    @cached_property
    def poloidal(self) -> TransportCoeff:
        return TransportCoeff(self["poloidal"], parent=self._parent)

    @cached_property
    def toroidal(self) -> TransportCoeff:
        return TransportCoeff(self["toroidal"], parent=self._parent)


class CoreTransportNeutral(Species):
    def __init__(self,   *args,  **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def ion_index(self):
        """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """
        return self.__raw_get__("ion_index")

    @cached_property
    def particles(self) -> TransportCoeff:
        return TransportCoeff(self["particles"], parent=self._parent)

    @cached_property
    def energy(self) -> TransportCoeff:
        return TransportCoeff(self["energy"],  parent=self._parent)


class CoreTransportProfiles1D(Profiles):
    Ion = CoreTransportIon
    Neutral = CoreTransportNeutral
    Electrons = CoreTransportElectrons
    Momentum = CoreTransportMomentum

    def __init__(self, *args, grid: RadialGrid = None,  **kwargs):
        super().__init__(*args,   **kwargs)
        self._grid = grid

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

    @cached_property
    def electrons(self) -> CoreTransportElectrons:
        """ Transport quantities related to the electrons """
        return CoreTransportProfiles1D.Electrons(self['electrons'], parent=self)

    @cached_property
    def ion(self) -> List[CoreTransportIon]:
        """ Transport coefficients related to the various ion species """
        return List[CoreTransportProfiles1D.Ion](self['ion'], parent=self)

    @cached_property
    def neutral(self) -> List[CoreTransportNeutral]:
        """ Transport coefficients related to the various neutral species """
        return List[CoreTransportProfiles1D.Neutral](self['neutral'],   parent=self)

    @cached_property
    def momentum(self) -> CoreTransportMomentum:
        return CoreTransportProfiles1D.Momentum(self["momentum"],  parent=self)

    @cached_property
    def total_ion_energy(self) -> TransportCoeff:
        """ Transport coefficients for the total (summed over ion species) energy equation """
        return TransportCoeff(self["total_ion_energy"], parent=self)

    @cached_property
    def momentum_tor(self) -> TransportCoeff:
        """ Transport coefficients for total toroidal momentum equation  """
        return TransportCoeff(self["momentum_tor"], parent=self)

    @cached_property
    def conductivity_parallel(self) -> Function:
        return Function(self.grid_d.rho_tor_norm, self["conductivity_parallel"])

    @cached_property
    def e_field_radial(self) -> Function:
        """ Radial component of the electric field (calculated e.g. by a neoclassical model) {dynamic} [V.m^-1]"""
        return Function(self.grid_flux.rho_tor_norm, self["e_field_radial"])


class CoreTransportTimeSlice(TimeSlice):

    def __init__(self,   *args, grid: RadialGrid = None,   **kwargs):
        super().__init__(*args, **kwargs)
        self._grid = grid

    @cached_property
    def profiles_1d(self) -> CoreTransportProfiles1D:
        return CoreTransportProfiles1D(self["profiles_1d"],  grid=self._grid,   parent=self)


class CoreTransportModel(Dict):
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
    TimeSlice = CoreTransportProfiles1D

    def __init__(self, *args,  grid: RadialGrid = None,   **kwargs):
        super().__init__(*args, **kwargs)
        self._grid = grid

    @cached_property
    def code(self) -> IDSCode:
        return IDSCode(self["code"])

    @cached_property
    def comment(self):
        return self["comment"]

    @cached_property
    def identifier(self) -> Identifier:
        r"""
            Transport model identifier. Available options (refer to the children of this identifier structure) :

            Name	            | Index	    | Description
            --------------------+-----------+-----------------------------------------------------------------------
            unspecified         | 0	        | Unspecified transport type
            combined	        | 1	        | Combination of data from available transport models. Representation of the total transport in the system
            transport_solver	| 2	        | Output from a transport solver
            background	        | 3	        | Background transport level, ad-hoc transport model not directly related to a physics model
            database	        | 4	        | Transport specified by a database entry external to the dynamic evolution of the plasma
            neoclassical	    | 5	        | Neoclassical
            anomalous           | 6	        | Representation of turbulent transport
            mhd                 | 19        | Transport arising from MHD frequency modes
            ntm                 | 20        | Transport arising from the presence of NTMs
            sawteeth            | 21        | Transport arising from the presence of sawteeth
            elm_continuous      | 22        | Continuous ELM model --- gives the ELM averaged profile
            elm_resolved        | 23        | Time resolved ELM model
            pedestal            | 24        | Transport level to give edge pedestal
            not_provided	    | 25        | No data provided

        """
        return Identifier(**self["identifier"]._as_dict())

    @cached_property
    def flux_multiplier(self) -> float:
        return self["flux_multiplier"]

    @cached_property
    def profiles_1d(self) -> TimeSeries[CoreTransportProfiles1D]:
        return TimeSeries[CoreTransportProfiles1D](self["profiles_1d"], parent=self)

    def update(self, d=None, *args, core_profiles=None,   grid=None, **kwargs):
        prof = d or {}
        # prof = d.setdefault("profiles_1d", {})
        if core_profiles is not None:
            prof["electrons"] = {}
            prof["ion"] = [
                {
                    "label": ion.label,
                    "z_ion": ion.z_ion,
                    "neutral_index": ion.neutral_index,
                    "element": ion.element._as_list(),
                }
                for ion in core_profiles.profiles_1d.ion
            ]
        logger.debug(prof)
        self.profiles_1d.insert(prof, *args,  grid=grid or self._grid,  **kwargs)


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
    _serialize_ignore = ["profiles_1d", ]
    Model = CoreTransportModel
    TimeSlice = CoreTransportTimeSlice

    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def model(self) -> List[CoreTransportModel]:
        return List[CoreTransportModel](self["model"], parent=self)

    def update(self, *args, **kwargs):
        for m in self.model:
            m.update(*args, **kwargs)

    @cached_property
    def profiles_1d(self) -> TimeSeries[CoreTransportProfiles1D]:
        return AttributeTree(Combiner(self.model, prefix="profiles_1d"), parent=self)
