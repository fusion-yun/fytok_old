import collections
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
from scipy import constants
from spdm.common.logger import logger
from spdm.common.tags import _undefined_
from spdm.data import Dict, File, Link, List, Node, Path, Query, sp_property, Function, Function

from ..common.IDS import IDS
from ..common.Misc import Decomposition, Identifier, VacuumToroidalField
from ..common.Module import Module
from ..common.Species import Species, SpeciesElectron, SpeciesIon
from ..transport.CoreProfiles import CoreProfiles
from ..transport.Equilibrium import Equilibrium
from ..transport.MagneticCoordSystem import RadialGrid

# class CoreSourcesParticle(Dict):
#     def __init__(self, *args,  **kwargs):
#         super().__init__(*args, **kwargs)

#     @sp_property
#     def particles(self):
#         return Function(self._parent.grid.rho_tor_norm, self["particles"]._parent)

#     @sp_property
#     def energy(self):
#         return Function(self._parent.grid.rho_tor_norm, self["energy"]._parent)

#     @sp_property
#     def momentum(self):
#         return Dict(self.get("momentum")._parent)

# {
#     "radial": Function(self._parent.radial_grid.rho_tor_norm, self["momentum.radial"]._parent),
#     "diamagnetic": Function(self._parent.radial_grid.rho_tor_norm, self["momentum.diamagnetic"]._parent),
#     "parallel": Function(self._parent.radial_grid.rho_tor_norm, self["momentum.parallel"]._parent),
#     "poloidal": Function(self._parent.radial_grid.rho_tor_norm, self["momentum.poloidal"]._parent),
#     "toroidal": Function(self._parent.radial_grid.rho_tor_norm, self["momentum.toroidal"]._parent)
# }


class CoreSourcesElectrons(SpeciesElectron):

    @sp_property
    def particles(self) -> Function:
        return self.get("particles", None)

    @sp_property
    def particles_decomposed(self) -> Decomposition[Function]:
        return self.get("particles_decomposed", None)

    @sp_property
    def energy(self) -> Function:
        return self.get("energy", None)

    @sp_property
    def energy_decomposed(self) -> Decomposition[Function]:
        return self.get("energy_decomposed", None)


class CoreSourcesIon(SpeciesIon):

    @sp_property
    def particles(self) -> Function:
        return self.get("particles")

    @sp_property
    def particles_fast(self) -> Function:
        """ NOT IN IMAS """
        return self.get("particles_fast")

    @sp_property
    def particles_decomposed(self) -> Decomposition[Function]:
        return self.get("particles_decomposed", {})

    @sp_property
    def energy(self) -> Function:
        return self.get("energy")

    @sp_property
    def energy_decomposed(self) -> Decomposition[Function]:
        return self.get("energy_decomposed", {})


class CoreSourcesNeutral(Species):
    pass


class CoreSourcesProfiles1D(Dict):
    Ion = CoreSourcesIon
    Electrons = CoreSourcesElectrons
    Neutral = CoreSourcesNeutral

    @sp_property
    def grid(self) -> RadialGrid:
        return self.get("grid")

    electrons: Electrons = sp_property()

    ion: List[Ion] = sp_property()

    neutral: List[Neutral] = sp_property()

    @sp_property
    def total_ion_energy(self):
        return Function(self.grid.rho_tor_norm, np.sum([ion.energy for ion in self.ion]))

    @sp_property
    def total_ion_power_inside(self) -> Function:
        return Function(self.grid.rho_tor_norm, self.get("total_ion_power_inside"))

    @sp_property
    def momentum_tor(self) -> Function:
        return Function(self.grid.rho_tor_norm, self.get("momentum_tor"))

    @sp_property
    def torque_tor_inside(self) -> Function:
        return Function(self.grid.rho_tor_norm, self.get("torque_tor_inside"))

    @sp_property
    def j_parallel(self) -> Function:
        return Function(self.grid.rho_tor_norm, self.get("j_parallel"))

    @sp_property
    def current_parallel_inside(self) -> Function:
        return Function(self.grid.rho_tor_norm, self.get("current_parallel_inside"))

    @sp_property
    def conductivity_parallel(self) -> Function:
        return Function(self.grid.rho_tor_norm, self.get("conductivity_parallel"))


class CoreSourcesGlobalQuantities(Dict):
    r"""
        Total source quantities integrated over the plasma volume or surface {dynamic}
    """

    pass


class CoreSourcesSpecies(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def type(self) -> Identifier:
        r"""
            Species type. index=1 for electron; index=2 for ion species in a single/average state (refer to ion structure); 
            index=3 for ion species in a particular state (refer to ion/state structure); 
            index=4 for neutral species in a single/average state (refer to neutral structure);
            index=5 for neutral species in a particular state (refer to neutral/state structure); 
            index=6 for neutron; index=7 for photon. Available options (refer to the children of this identifier structure) :

           +---------------+------------+-------------------------------------------------------------------------+
           |Name           | Index      |  Description                                                            |                                                                         
           +===============+============+=========================================================================+ 
           |unspecified	   | 0	        |  unspecified                                                            |           
           +---------------+------------+-------------------------------------------------------------------------+                                                                         
           |electron	   | 1	        |  Electron                                                               |        
           +---------------+------------+-------------------------------------------------------------------------+                                                                         
           |ion	           | 2	        |  Ion species in a single/average state; refer to ion-structure          |                                                             
           +---------------+------------+-------------------------------------------------------------------------+                                                                         
           |ion_state	   | 3          |  Ion species in a particular state; refer to ion/state-structure        |                                                               
           +---------------+------------+-------------------------------------------------------------------------+                                                                         
           |neutral        | 4	        |  Neutral species in a single/average state; refer to neutral-structure  |                                                                     
           +---------------+------------+-------------------------------------------------------------------------+                                                                         
           |neutral_state  | 5	        |  Neutral species in a particular state; refer to neutral/state-structure|                                                                       
           +---------------+------------+-------------------------------------------------------------------------+                                                                         
           |neutron        | 6	        |  Neutron                                                                |       
           +---------------+------------+-------------------------------------------------------------------------+                                                                         
           |photon         | 7	        |  Photon                                                                 |
           +---------------+------------+-------------------------------------------------------------------------+

           
        """
        return self.get("type", {})


class CoreSourcesSource(Module):
    r"""
            Source term identifier (process causing this source term). Available options (refer to the children of this identifier structure) :

            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | Name                                       | Index            | Description                                                                                      |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | unspecified                                |  0	            | Unspecified source type                                                                          |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | total                                      |  1	            | Total source; combines all sources                                                               |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | nbi                                        |  2	            | Source from Neutral Beam Injection                                                               |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | ec                                         |  3	            | Sources from electron cyclotron heating and current drive                                        |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | lh                                         |  4	            | Sources from lower hybrid heating and current drive                                              |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | ic                                         |  5	            | Sources from heating at the ion cyclotron range of frequencies                                   |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | fusion	                                 |  6	            | Sources from fusion reactions, e.g. alpha particle heating                                       |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | ohmic                                      |  7	            | Source from ohmic heating                                                                        |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | bremsstrahlung                             |  8	            | Source from bremsstrahlung; radiation losses are negative sources                                |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | synchrotron_radiation                      |  9	            | Source from synchrotron radiation; radiation losses are negative sources                         |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | line_radiation                             | 10	            | Source from line radiation; radiation losses are negative sources                                |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | collisional_equipartition                  | 11	            | Collisional equipartition                                                                        |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | cold_neutrals                              | 12	            | Source of cold neutrals                                                                          |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | bootstrap_current                          | 13	            | Bootstrap current                                                                                |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | pellet                                     | 14	            | Sources from injection                                                                           |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | auxiliary                                  | 100              | Source from auxiliary systems, e.g. heating and current drive systems                            |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | ic_nbi                                     | 101              | A combination of the ic and nbi sources                                                          |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | ic_fusion                                  | 102              | A combination of the ic and fusion sources                                                       |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | ic_nbi_fusion                              | 103              | A combination of the ic and fusion sources                                                       |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | ec_lh                                      | 104              | A combination of the ec and lh sources                                                           |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | ec_ic                                      | 105              | A combination of the ec and ic sources                                                           |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | lh_ic                                      | 106              | A combination of the lh and ic sources                                                           |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | ec_lh_ic                                   | 107              | A combination of the ec, lh and ic sources                                                       |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | gas_puff                                   | 108              | Gas puff                                                                                         |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | killer_gas_puff                            | 109              | Killer gas puff                                                                                  |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | radiation	                                 | 200              | Total radiation source; radiation losses are negative sources                                    |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | cyclotron_radiation                        | 201              | Source from cyclotron radiation; radiation losses are negative sources                           |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | cyclotron_synchrotron_radiation            | 202              | Source from combined cyclotron and synchrotron radiation; radiation losses are negative sources  |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | impurity_radiation                         | 203              | Line radiation and Bremsstrahlung source; radiation losses are negative sources.                 |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | particles_to_wall                          | 303              | Particle pumping by the wall; negative source for plasma and positive source for the wall        |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | particles_to_pump                          | 304              | Particle pumping by external pump; negative source for plasma and positive source for the pump   |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | charge_exchange                            | 305              | Source from charge exchange. Charge exchange losses are negative sources                         |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | transport                                  | 400              | Source term related to transport processes                                                       |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | neoclassical                               | 401              | Source term related to neoclassical processes                                                    |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | equipartition                              | 402              | Equipartition due to collisions and turbulence                                                   |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | turbulent_equipartition                    | 403              | Turbulent equipartition                                                                          |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | runaways                                   | 501              | Source from run-away processes; includes both electron and ion run-away                          |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | ionisation                                 | 601              | Source from ionisation processes (not accounting for charge exchange)                            |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | recombination                              | 602              | Source from recombination processes (not accounting for charge exchange)                         |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | excitation                                 | 603              | Source from excitation processes                                                                 |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | database                                   | 801              | Source from database entry                                                                       |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | gaussian                                   | 802              | Artificial source with a gaussian profile                                                        |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | custom_1                                   | 901              | Custom source terms 1; content to be decided by data provided                                    |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | custom_2                                   | 902              | Custom source terms 2; content to be decided by data provided                                    |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | custom_3                                   | 903              | Custom source terms 3; content to be decided by data provided                                    |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | custom_4                                   | 904              | Custom source terms 4; content to be decided by data provided                                    |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | custom_5                                   | 905              | Custom source terms 5; content to be decided by data provided                                    |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | custom_6                                   | 906              | Custom source terms 6; content to be decided by data provided                                    |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | custom_7                                   | 907              | Custom source terms 7; content to be decided by data provided                                    |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | custom_8                                   | 908              | Custom source terms 8; content to be decided by data provided                                    |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+
            | custom_9                                   | 909              | Custom source terms 9; content to be decided by data provided                                    |
            +--------------------------------------------+------------------+--------------------------------------------------------------------------------------------------+

        """

    _fy_module_prefix = "fymodules.transport.core_sources."

    Species = CoreSourcesSpecies
    Profiles1D = CoreSourcesProfiles1D
    GlobalQuantities = CoreSourcesGlobalQuantities

    species: Species = sp_property()

    global_quantities: GlobalQuantities = sp_property()

    profiles_1d: Profiles1D = sp_property()

    def refresh(self, *args, core_profiles: CoreProfiles,  **kwargs) -> float:
        residual = super().refresh(*args,  **kwargs)
        self.profiles_1d["grid"] = core_profiles.profiles_1d.grid
        return residual


class CoreSources(IDS):
    """CoreSources
    """
    _IDS = "core_sources"
    Source = CoreSourcesSource

    vacuum_toroidal_field: VacuumToroidalField = sp_property()

    source: List[Source] = sp_property()

    @cached_property
    def source_combiner(self) -> Source:
        return self.source.combine({
            "identifier": {"name": "total", "index": 1,
                           "description": "Total source; combines all sources"},
            "code": {"name": None}
        })

    def refresh(self, *args,   **kwargs) -> float:
        if "source_combiner" in self.__dict__:
            del self.__dict__["source_combiner"]

        for src in self.source:
            src.refresh(*args, **kwargs)
