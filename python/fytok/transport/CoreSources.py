from functools import cached_property

import numpy as np
from spdm.data.Dict import Dict
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.Function import Function
from spdm.data.sp_property import sp_property

from ..common.IDS import IDS
from ..common.Misc import Decomposition, Identifier, VacuumToroidalField
from ..common.Module import Module
from .CoreProfiles import CoreProfiles
from .MagneticCoordSystem import RadialGrid
from .Species import Species, SpeciesElectron, SpeciesIon

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

    particles: Function = sp_property()

    particles_decomposed: Decomposition[Function] = sp_property()

    energy: Function = sp_property()

    energy_decomposed: Decomposition[Function] = sp_property()


class CoreSourcesIon(SpeciesIon):

    particles: Function = sp_property()

    particles_fast: Function = sp_property()

    particles_decomposed: Decomposition[Function] = sp_property()

    energy: Function = sp_property()

    energy_decomposed: Decomposition[Function] = sp_property()


class CoreSourcesNeutral(Species):
    pass


class CoreSourcesProfiles1D(Dict[Node]):
    Ion = CoreSourcesIon
    Electrons = CoreSourcesElectrons
    Neutral = CoreSourcesNeutral

    @property
    def grid(self) -> RadialGrid:
        return self._parent.grid

    electrons: Electrons = sp_property()

    ion: List[Ion] = sp_property()

    neutral: List[Neutral] = sp_property()

    @sp_property
    def total_ion_energy(self) -> Function:
        return np.sum([ion.energy for ion in self.ion])

    total_ion_power_inside: Function = sp_property()

    momentum_tor: Function = sp_property()

    torque_tor_inside: Function = sp_property()

    j_parallel: Function = sp_property()

    current_parallel_inside: Function = sp_property()

    conductivity_parallel: Function = sp_property()


class CoreSourcesGlobalQuantities(Dict[Node]):
    r"""
        Total source quantities integrated over the plasma volume or surface {dynamic}
    """
    pass


class CoreSourcesSpecies(Species):

    type: Identifier = sp_property()
    """
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

    @property
    def grid(self):
        return self._parent.grid

    Species = CoreSourcesSpecies

    Profiles1D = CoreSourcesProfiles1D

    GlobalQuantities = CoreSourcesGlobalQuantities

    species: Species = sp_property()

    global_quantities: GlobalQuantities = sp_property()

    profiles_1d: Profiles1D = sp_property()

    def refresh(self, *args, **kwargs) -> float:
        residual = super().refresh(*args,  **kwargs)
        return residual


class CoreSources(IDS):
    """
            Core plasma thermal source terms (for the transport equations of the thermal species).
            Energy terms correspond to the full kinetic energy equation
            (i.e. the energy flux takes into account the energy transported by the particle flux)
    """
    _IDS = "core_sources"
    
    Source = CoreSourcesSource

    vacuum_toroidal_field: VacuumToroidalField = sp_property()

    grid: RadialGrid = sp_property()

    source: List[Source] = sp_property()

    @property
    def source_combiner(self) -> Source:
        return self.source.combine(
            common_data={
                "identifier": {"name": "total", "index": 1,
                               "description": "Total source; combines all sources"},
                "code": {"name": None},
            })

    def refresh(self,  core_profiles: CoreProfiles, *args, **kwargs) -> float:
        self["grid"] = core_profiles.profiles_1d.grid
        for src in self.source:
            src.refresh(*args, core_profiles=core_profiles, **kwargs)
