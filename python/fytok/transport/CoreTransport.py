import collections
from dataclasses import dataclass
from functools import cached_property
from typing import ChainMap, Optional

import numpy as np
from importlib_metadata import metadata
from spdm.common.tags import _not_found_, _undefined_
from spdm.data import (Dict, File, Function, Link, List, Node, Path, Query,
                       function_like, sp_property)
from spdm.util.logger import logger

from ..common.IDS import IDS, IDSCode
from ..common.Misc import Identifier, VacuumToroidalField
from ..common.Module import Module
from .CoreProfiles import CoreProfiles
from .Equilibrium import Equilibrium
from .MagneticCoordSystem import RadialGrid
from .Species import Species, SpeciesElectron, SpeciesIon, SpeciesIonState


class TransportCoeff(Dict):
    d: Function = sp_property(lambda self: function_like(self._parent._parent.grid_d.rho_tor_norm, self.get("d", 0)))

    v: Function = sp_property(lambda self: function_like(self._parent._parent.grid_v.rho_tor_norm, self.get("v", 0)))

    flux: Function = sp_property(
        lambda self: function_like(self._parent._parent.grid_flux.rho_tor_norm, self.get("flux", 0)))

    d_fast_factor: Function = sp_property(lambda self: function_like(
        self._parent._parent.grid.rho_tor_norm, self.get("d_fast_factor", 1)))
    """ NOT IN IMAS """

    v_fast_factor: Function = sp_property(lambda self: function_like(
        self._parent._parent.grid.rho_tor_norm, self.get("v_fast_factor", 1)))
    """ NOT IN IMAS """

    flux_fast_factor: Function = sp_property(lambda self:  function_like(
        self._parent._parent.grid_flux.rho_tor_norm, self.get("flux_fast", 1)))
    """ NOT IN IMAS """


class CoreTransportElectrons(SpeciesElectron):

    particles: TransportCoeff = sp_property()

    energy: TransportCoeff = sp_property()


class CoreTransportIonState(SpeciesIonState):

    particles: TransportCoeff = sp_property()
    """Transport quantities related to density equation of the charge state considered (thermal+non-thermal)	structure	"""

    energy: TransportCoeff = sp_property()
    """Transport quantities related to the energy equation of the charge state considered	structure	"""

    momentum: TransportCoeff = sp_property()
    """Transport coefficients related to the state momentum equations for various components (directions)"""


class CoreTransportMomentum(Dict):

    radial: TransportCoeff = sp_property()

    diamagnetic: TransportCoeff = sp_property()

    parallel: TransportCoeff = sp_property()

    poloidal: TransportCoeff = sp_property()

    toroidal: TransportCoeff = sp_property()


class CoreTransportIon(SpeciesIon):

    particles: TransportCoeff = sp_property()

    energy: TransportCoeff = sp_property()

    momentum: CoreTransportMomentum = sp_property()

    state: List[CoreTransportIonState] = sp_property()


class CoreTransportNeutral(Species):

    ion_index: int = sp_property(default_value=0)
    """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """

    particles: TransportCoeff = sp_property()

    energy: TransportCoeff = sp_property()


class CoreTransportProfiles1D(Dict[Node]):
    Ion = CoreTransportIon
    Neutral = CoreTransportNeutral
    Electrons = CoreTransportElectrons
    Momentum = CoreTransportMomentum

    grid: RadialGrid = sp_property()

    # grid_d: RadialGrid = sp_property(
    #     lambda self: self.grid.remesh("rho_tor_norm", 0.5*(self.grid.rho_tor_norm[:-1]+self.grid.rho_tor_norm[1:])),
    #     doc="""Grid for effective diffusivity and parallel conductivity""")

    @sp_property
    def grid_d(self) -> RadialGrid:
        return self.grid.remesh("rho_tor_norm", 0.5*(self.grid.rho_tor_norm[:-1]+self.grid.rho_tor_norm[1:]))

    grid_v: RadialGrid = sp_property(lambda self: self.grid.remesh("rho_tor_norm", self.grid.rho_tor_norm))
    """ Grid for effective convections  """

    grid_flux: RadialGrid = sp_property(lambda self:
                                        self.grid.remesh("rho_tor_norm", 0.5*(self.grid.rho_tor_norm[:-1]+self.grid.rho_tor_norm[1:])))
    """ Grid for fluxes  """

    electrons: Electrons = sp_property()
    """ Transport quantities related to the electrons """

    ion: List[Ion] = sp_property()
    """ Transport coefficients related to the various ion species """

    neutral: List[Neutral] = sp_property()
    """ Transport coefficients related to the various neutral species """

    momentum: Momentum = sp_property()

    total_ion_energy: TransportCoeff = sp_property()
    """ Transport coefficients for the total (summed over ion species) energy equation """

    momentum_tor: TransportCoeff = sp_property()
    """ Transport coefficients for total toroidal momentum equation  """

    conductivity_parallel: Function = sp_property(lambda self: function_like(
        self.grid_d.rho_tor_norm, self.get("conductivity_parallel")))

    e_field_radial: Function = sp_property(lambda self:  function_like(
        self.grid_flux.rho_tor_norm, self.get("e_field_radial")))
    """ Radial component of the electric field (calculated e.g. by a neoclassical model) {dynamic} [V.m^-1]"""


class CoreTransportModel(Module):
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

    _fy_module_prefix = "fymodules.transport.core_transport."

    Profiles1D = CoreTransportProfiles1D

    grid: RadialGrid = sp_property()

    flux_multiplier: float = sp_property(default_value=1.0)

    profiles_1d: Profiles1D = sp_property()

    def refresh(self, *args, core_profiles: CoreProfiles, **kwargs) -> None:
        super().refresh(*args, core_profiles=core_profiles, **kwargs)
        self.profiles_1d["grid"] = core_profiles.profiles_1d.grid


class CoreTransport(IDS):
    r"""
        Core plasma transport of particles, energy, momentum and poloidal flux. The transport of particles, energy and momentum is described by
        diffusion coefficients,  $D$, and convection velocities,  $v$. These are defined by the total fluxes of particles, energy and momentum, across a
        flux surface given by : $V^{\prime}\left[-DY^{\prime}\left|\nabla\rho_{tor,norm}\right|^{2}+vY\left|\nabla\rho_{tor,norm}\right|\right]$,
        where $Y$ represents the particles, energy and momentum density, respectively, while  $V$ is the volume inside a flux surface, the primes denote
        derivatives with respect to $\rho_{tor,norm}$ and
        $\left\langle X\right\rangle$ is the flux surface average of a quantity  $X$. This formulation remains valid when changing simultaneously
        $\rho_{tor,norm}$ into $rho_{tor}$
        in the gradient terms and in the derivatives denoted by the prime. The average flux stored in the IDS as sibling of  $D$ and  $v$ is the total
        flux described above divided by the flux surface area $V^{\prime}\left\langle \left|\nabla\rho_{tor,norm}\right|\right\rangle$ .
        Note that the energy flux includes the energy transported by the particle flux.
    """
    _IDS = "core_transport"
    Model = CoreTransportModel

    vacuum_toroidal_field: VacuumToroidalField = sp_property()

    grid: RadialGrid = sp_property()

    model: List[Model] = sp_property()

    @cached_property
    def model_combiner(self) -> Model:
        return self.model.combine({
            "identifier": {"name": "combined", "index": 1,
                           "description": """Combination of data from available transport models.
                                Representation of the total transport in the system"""},
            "code": {"name": None},
            # "profiles_1d": {"grid": self.grid}
        })

    def refresh(self, *args,   **kwargs) -> None:
        if "model_combiner" in self.__dict__:
            del self.__dict__["model_combiner"]
        for model in self.model:
            model.refresh(*args, **kwargs)
