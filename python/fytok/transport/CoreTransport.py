import collections
from dataclasses import dataclass
from typing import Optional

from spdm.data.Function import Function, function_like
from spdm.data.Node import Dict, List, Node, sp_property
from spdm.data.Profiles import Profiles
from spdm.flow.Actor import Actor
from spdm.numlib import np
from spdm.util.logger import logger
from spdm.util.utilities import _not_found_

from ..common.IDS import IDS, IDSCode
from ..common.Misc import Identifier, VacuumToroidalField
from ..common.Species import (Species, SpeciesElectron, SpeciesIon,
                              SpeciesIonState)
from .CoreProfiles import CoreProfiles
from .Equilibrium import Equilibrium
from .MagneticCoordSystem import RadialGrid


class TransportCoeff(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,  ** kwargs)

    @sp_property
    def d(self) -> Function:
        return function_like(self._parent._grid.rho_tor_norm, self.get("d", None))

    @sp_property
    def v(self) -> Function:
        return function_like(self._parent._grid.rho_tor_norm,  self.get("v", None))

    @sp_property
    def flux(self) -> Function:
        return function_like(self._parent._grid.rho_tor_norm, self.get("flux", None))


class CoreTransportElectrons(SpeciesElectron):
    def __init__(self,   *args,  **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def particles(self) -> TransportCoeff:
        return self.get("particles", {})

    @sp_property
    def energy(self) -> TransportCoeff:
        return self.get("energy", {})


class CoreTransportIonState(SpeciesIonState):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @sp_property
    def particles(self) -> TransportCoeff:
        """Transport quantities related to density equation of the charge state considered (thermal+non-thermal)	structure	"""
        return self.get("particles", {})

    @sp_property
    def energy(self) -> TransportCoeff:
        """Transport quantities related to the energy equation of the charge state considered	structure	"""
        return self.get("energy", {})

    @sp_property
    def momentum(self) -> TransportCoeff:
        """Transport coefficients related to the state momentum equations for various components (directions)"""
        return self.get("momentum", {})


class CoreTransportMomentum(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,  **kwargs)

    @sp_property
    def radial(self) -> TransportCoeff:
        return TransportCoeff(self["radial"], parent=self._parent)

    @sp_property
    def diamagnetic(self) -> TransportCoeff:
        return TransportCoeff(self["diamagnetic"], parent=self._parent)

    @sp_property
    def parallel(self) -> TransportCoeff:
        return TransportCoeff(self["parallel"], parent=self._parent)

    @sp_property
    def poloidal(self) -> TransportCoeff:
        return TransportCoeff(self["poloidal"], parent=self._parent)

    @sp_property
    def toroidal(self) -> TransportCoeff:
        return TransportCoeff(self["toroidal"], parent=self._parent)


class CoreTransportIon(SpeciesIon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def particles(self) -> TransportCoeff:
        return TransportCoeff(self["particles"], parent=self._parent)

    @sp_property
    def energy(self) -> TransportCoeff:
        return TransportCoeff(self["energy"], parent=self._parent)

    @sp_property
    def momentum(self) -> CoreTransportMomentum:
        return self["momentum"]

    @sp_property
    def state(self) -> List[CoreTransportIonState]:
        return self["state"]


class CoreTransportNeutral(Species):
    def __init__(self,   *args,  **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def ion_index(self) -> int:
        """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """
        return self["ion_index"]

    @sp_property
    def particles(self) -> TransportCoeff:
        return TransportCoeff(self["particles"], parent=self._parent)

    @sp_property
    def energy(self) -> TransportCoeff:
        return TransportCoeff(self["energy"], parent=self._parent)


class CoreTransportProfiles1D(Dict[Node]):
    Ion = CoreTransportIon
    Neutral = CoreTransportNeutral
    Electrons = CoreTransportElectrons
    Momentum = CoreTransportMomentum

    def __init__(self, *args, grid: Optional[RadialGrid] = None, parent=None, ** kwargs):
        if grid is None:
            grid = parent._grid
        super().__init__(*args, axis=grid.rho_tor_norm,   **kwargs)
        self._grid = grid

    @property
    def grid(self) -> RadialGrid:
        return self._grid

    @sp_property
    def grid_d(self) -> RadialGrid:
        """Grid for effective diffusivities and parallel conductivity"""
        return self._grid.remesh(0.5*(self._grid.psi_norm[:-1]+self._grid.psi_norm[1:]))

    @sp_property
    def grid_v(self) -> RadialGrid:
        """ Grid for effective convections  """
        return self._grid.remesh(self._grid.psi_norm)

    @sp_property
    def grid_flux(self) -> RadialGrid:
        """ Grid for fluxes  """
        return self._grid.remesh(0.5*(self._grid.psi_norm[:-1]+self._grid.psi_norm[1:]))

    @sp_property
    def electrons(self) -> CoreTransportElectrons:
        """ Transport quantities related to the electrons """
        res = self.get('electrons', {})

        return res

    @sp_property
    def ion(self) -> List[CoreTransportIon]:
        """ Transport coefficients related to the various ion species """
        return self.get('ion', [])

    @sp_property
    def neutral(self) -> List[CoreTransportNeutral]:
        """ Transport coefficients related to the various neutral species """
        return self.get('neutral', [])

    @sp_property
    def momentum(self) -> CoreTransportMomentum:
        return self.get('momentum', {})

    @sp_property
    def total_ion_energy(self) -> TransportCoeff:
        """ Transport coefficients for the total (summed over ion species) energy equation """
        return self.get("total_ion_energy", {})

    @sp_property
    def momentum_tor(self) -> TransportCoeff:
        """ Transport coefficients for total toroidal momentum equation  """
        return self.get("momentum_tor", {})

    @sp_property
    def conductivity_parallel(self) -> Function:
        return function_like(self.grid_d.rho_tor_norm, self.get("conductivity_parallel", 0))

    @sp_property
    def e_field_radial(self) -> Function:
        """ Radial component of the electric field (calculated e.g. by a neoclassical model) {dynamic} [V.m^-1]"""
        return function_like(self.grid_flux.rho_tor_norm, self.get("e_field_radial", 0))


class CoreTransportModel(Actor):

    _actor_module_prefix = "transport.core_transport."

    Profiles1D = CoreTransportProfiles1D

    def __init__(self, *args, grid: Optional[RadialGrid] = None, ** kwargs):
        super().__init__(*args, **kwargs)
        self._grid = grid or getattr(self._parent, "_grid", None)
        self._equilibrium = getattr(self._parent, "equilibrium", None)
        self._core_profiles = getattr(self._parent, "core_profiles", None)

    @sp_property
    def code(self) -> IDSCode:
        return self.get("code", {})

    @sp_property
    def comment(self) -> str:
        return self.get("comment", "")

    @sp_property
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

    @sp_property
    def flux_multiplier(self) -> float:
        return self.get("flux_multiplier", 1.0)

    @sp_property
    def profiles_1d(self) -> CoreTransportProfiles1D:
        return self.get("profiles_1d", {})

    def update(self,  *args, grid=None, equilibrium: Equilibrium = None, core_profiles: CoreProfiles = None,  **kwargs) -> float:
        time = super().update(*args, **kwargs)
        if grid is not None:
            self._grid = grid
        if equilibrium is not None:
            self._equilibrium = equilibrium
        if core_profiles is not None:
            self._core_profiles = core_profiles

        super().reset()
        return time


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
    Model = CoreTransportModel

    def __init__(self, *args, grid: RadialGrid = None, ** kwargs):
        super().__init__(*args,  **kwargs)
        self._grid = grid or getattr(self._parent, "grid", _not_found_)

    @sp_property
    def vacuum_toroidal_field(self) -> VacuumToroidalField:
        return VacuumToroidalField(**self.get("vacuum_toroidal_field", {}))

    @sp_property
    def model(self) -> List[Model]:
        return List[CoreTransport.Model](
            self.get("model", []),
            defualt_value_when_combine={
                "identifier": {"name": "combined", "index": 1,
                               "description": """Combination of data from available transport models.
                                Representation of the total transport in the system"""}
            },
            parent=self)

    def update(self, *args, **kwargs) -> None:
        self.model.update(*args, **kwargs)
