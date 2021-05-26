import collections
from typing import Optional

from spdm.util.numlib import np
from spdm.util.numlib import constants
from spdm.data.Function import Function
from spdm.data.Node import Dict, List, Node
from spdm.data.Profiles import Profiles
from spdm.data.Node import sp_property
from spdm.flow.Actor import Actor
from spdm.util.logger import logger

from ..common.IDS import IDS
from ..common.Misc import Identifier, VacuumToroidalField
from ..common.Species import Species
from .MagneticCoordSystem import RadialGrid


class CoreSourcesParticle(Dict):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def particles(self):
        return Function(self._parent.grid.rho_tor_norm, self["particles"], parent=self._parent)

    @sp_property
    def energy(self):
        return Function(self._parent.grid.rho_tor_norm, self["energy"], parent=self._parent)

    @sp_property
    def momentum(self):
        return Profiles(self["momentum"], axis=self._parent.grid.rho_tor_norm, parent=self._parent)

        # {
        #     "radial": Function(self._parent.grid.rho_tor_norm, self["momentum.radial"], parent=self._parent),
        #     "diamagnetic": Function(self._parent.grid.rho_tor_norm, self["momentum.diamagnetic"], parent=self._parent),
        #     "parallel": Function(self._parent.grid.rho_tor_norm, self["momentum.parallel"], parent=self._parent),
        #     "poloidal": Function(self._parent.grid.rho_tor_norm, self["momentum.poloidal"], parent=self._parent),
        #     "toroidal": Function(self._parent.grid.rho_tor_norm, self["momentum.toroidal"], parent=self._parent)
        # }


class CoreSourcesElectrons(Dict):
    def __init__(self,  *args,   **kwargs):
        super().__init__(* args,  **kwargs)

    @sp_property
    def particle(self):
        return (
            + self.S_neutrals   # ionization source from neutrals (wall recycling, gas puffing, etc),
            + self.S_nbi        # NBI,
            + self.S_ext        # optional additional source ‘EXT’,
            + self.S_ripple     # particle losses induced by toroidal magnetic field ripple.
        )

    @sp_property
    def energy(self):
        return (
            - self._parent.Qei      # electron–ion collisional energy transfer
            + self._parent.Qneo     # neoclassical contribution
            # + self.Qoh              # ohmic
            # + self.Qe_lh            # LH,
            # + self.Qe_nbi           # NBI,
            # + self.Qe_icrh          # ICRH,
            # + self.Qe_ecrh          # ECRH
            # + self.Qe_n0            # charge exchange
            # + self.Qe_ext           # optional additional source ‘EXT’
            # - self.Qrad             # line radiation
            # - self.Qbrem            # bremsstrahlung
            # - self.Qcyclo           # synchroton radiation
            # + self.Qe_fus           # fusion reactions
            # + self.Qe_rip           # energy losses induced by toroidal magnetic field ripple
        )


class CoreSourcesIon(Dict):
    def __init__(self,  *args,   **kwargs):
        super().__init__(* args,  **kwargs)

    @sp_property
    def energy(self):
        return (
            + self._parent.Qei      # electron–ion collisional energy transfer
            - self._parent.Qneo     # neoclassical contribution (opposite sign w.r.t. electron heat equation)
            + self.Qi_lh            # LH,
            + self.Qi_nbi           # NBI,
            + self.Qi_icrh          # ICRH,
            + self.Qi_ecrh          # ECRH
            + self.Qi_n0            # charge exchange
            + self.Qi_ext           # optional additional source EXT’
            + self.Qi_fus           # fusion reactions
            + self.Qi_rip           # energy losses induced by toroidal magnetic field ripple
        )


class CoreSourcesNeutral(Profiles):
    def __init__(self,  *args,   **kwargs):
        super().__init__(* args,  **kwargs)


class CoreSourcesProfiles1D(Profiles):
    def __init__(self, *args, grid: RadialGrid = None, **kwargs):
        super().__init__(*args,  **kwargs)
        self._grid = grid or self._parent._grid

    @property
    def time(self) -> float:
        return self._time

    @property
    def grid(self) -> RadialGrid:
        return self._grid

    @sp_property
    def electrons(self) -> CoreSourcesElectrons:
        return self["electrons"]

    @sp_property
    def ion(self) -> List[CoreSourcesIon]:
        return self["ion"]

    @sp_property
    def neutral(self) -> List[CoreSourcesNeutral]:
        return self["neutral"]

    @sp_property
    def total_ion_energy(self):
        res = Function(self.grid.rho_tor_norm,  0.0)
        for ion in self.ion:
            res += ion.energy
        return res

    @sp_property
    def total_ion_power_inside(self):
        return NotImplemented

    @sp_property
    def torque_tor_inside(self):
        return NotImplemented

    @sp_property
    def j_parallel(self):
        return Function(self.grid.rho_tor_norm, self.get("j_parallel", None))

    @sp_property
    def current_parallel_inside(self):
        return Function(self.grid.rho_tor_norm, self["current_parallel_inside"])

    @sp_property
    def conductivity_parallel(self):
        return Function(self.grid.rho_tor_norm, self["conductivity_parallel"])

    @sp_property
    def Qei(self):
        Te = self._core_profile.profiles_1d.electrons.temperature
        ne = self._core_profile.profiles_1d.electrons.density

        gamma_ei = 15.2 - np.log(ne)/np.log(1.0e20) + np.log(Te)/np.log(1.0e3)
        epsilon = constants.epsilon_0
        e = constants.elementary_charge
        me = constants.electron_mass
        mp = constants.proton_mass
        PI = constants.pi
        tau_e = 12*(PI**(3/2))*(epsilon**2)/(e**4)*np.sqrt(me/2)*((e*Te)**(3/2))/ne/gamma_ei

        def qei(ion):
            return ion.density*(ion.z_ion**2)/sum(ele.atoms_n*ele.a for ele in ion.element)*(Te-ion.temperature)

        return sum(qei(ion) for ion in self._core_profile.ions)*(3/2) * e/(mp/me/2)/tau_e

    @sp_property
    def Qneo(self):
        return NotImplemented

    @sp_property
    def Qoh(self):
        return NotImplemented

    @sp_property
    def j_ni(self):
        r"""
            the current density driven by the non-inductive sources
        """
        return (
            self.j_boot         # bootstrap current
            + self.j_nbi        # neutral beam injection (NBI)
            + self.j_lh         # lower hybrid (LH) waves
            + self.j_ec         # electron cyclotron (EC) waves
            + self.j_ic         # ion cyclotron (IC) waves
            + self.j_ext        # current source ‘EXT
        )


class CoreSourcesGlobalQuantities(Dict):
    r"""
        Total source quantities integrated over the plasma volume or surface {dynamic}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CoreSourcesSpecies(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def type(self) -> Identifier:
        r"""
            Species type. index=1 for electron; index=2 for ion species in a single/average state (refer to ion structure); index=3 for ion species in a particular state (refer to ion/state structure); index=4 for neutral species in a single/average state (refer to neutral structure); index=5 for neutral species in a particular state (refer to neutral/state structure); index=6 for neutron; index=7 for photon. Available options (refer to the children of this identifier structure) :

            Name	       | Index      |  Description
            ---------------+------------+----------------------------------------------------------------
            unspecified	   | 0	        |  unspecified
            electron	   | 1	        |  Electron
            ion	           | 2	        |  Ion species in a single/average state; refer to ion-structure
            ion_state	   | 3          |  Ion species in a particular state; refer to ion/state-structure
            neutral	       | 4	        |  Neutral species in a single/average state; refer to neutral-structure
            neutral_state  | 5	        |  Neutral species in a particular state; refer to neutral/state-structure
            neutron        | 6	        |  Neutron
            photon         | 7	        |  Photon
        """
        return Identifier(**self["type"]._as_dict())


# class CoreSourcesTimeSlice(Dict):
#     GlobalQuantities = CoreSourcesGlobalQuantities
#     Profiles1D = CoreSourcesProfiles1D

#     def __init__(self,   *args, grid=None, time=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._grid = grid or self._parent._grid
#         self._time = time or self._parent

#     @sp_property
#     def vacuum_toroidal_field(self):
#         return VacuumToroidalField(**self["vacuum_toroidal_field"]._as_dict())

#     @sp_property
#     def global_quantities(self) -> GlobalQuantities:
#         return CoreSources.GlobalQuantities(self["global_quantities"], time=self._time,  parent=self)

#     @sp_property
#     def profiles_1d(self) -> Profiles1D:
#         return CoreSources.Profiles1D(self["profiles_1d"], grid=self._grid,  time=self._time, parent=self)


class CoreSourcesSource(Actor):
    _actor_module_prefix = "transport.core_sources."

    def __init__(self,   *args, grid: Optional[RadialGrid] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._grid = grid or self._parent._grid

    @sp_property
    def identifier(self) -> Identifier:
        r"""
            Source term identifier (process causing this source term). Available options (refer to the children of this identifier structure) :

            Name	                                   | Index	        | Description
            -------------------------------------------+----------------+-----------------------------------------------------------------------
            unspecified                                |  0	            | Unspecified source type
            total                                      |  1	            | Total source; combines all sources
            nbi                                        |  2	            | Source from Neutral Beam Injection
            ec                                         |  3	            | Sources from electron cyclotron heating and current drive
            lh                                         |  4	            | Sources from lower hybrid heating and current drive
            ic                                         |  5	            | Sources from heating at the ion cyclotron range of frequencies
            fusion	                                   |  6	            | Sources from fusion reactions, e.g. alpha particle heating
            ohmic	                                   |  7	            | Source from ohmic heating
            bremsstrahlung                             |  8	            | Source from bremsstrahlung; radiation losses are negative sources
            synchrotron_radiation                      |  9	            | Source from synchrotron radiation; radiation losses are negative sources
            line_radiation                             | 10	            | Source from line radiation; radiation losses are negative sources
            collisional_equipartition                  | 11	            | Collisional equipartition
            cold_neutrals                              | 12	            | Source of cold neutrals
            bootstrap_current                          | 13	            | Bootstrap current
            pellet                                     | 14	            | Sources from injection
            auxiliary                                  | 100            | Source from auxiliary systems, e.g. heating and current drive systems
            ic_nbi                                     | 101            | A combination of the ic and nbi sources
            ic_fusion                                  | 102            | A combination of the ic and fusion sources
            ic_nbi_fusion                              | 103            | A combination of the ic and fusion sources
            ec_lh                                      | 104            | A combination of the ec and lh sources
            ec_ic                                      | 105            | A combination of the ec and ic sources
            lh_ic                                      | 106            | A combination of the lh and ic sources
            ec_lh_ic                                   | 107            | A combination of the ec, lh and ic sources
            gas_puff                                   | 108            | Gas puff
            killer_gas_puff                            | 109            | Killer gas puff
            radiation	                               | 200            | Total radiation source; radiation losses are negative sources
            cyclotron_radiation                        | 201            | Source from cyclotron radiation; radiation losses are negative sources
            cyclotron_synchrotron_radiation            | 202            | Source from combined cyclotron and synchrotron radiation; radiation losses are negative sources
            impurity_radiation                         | 203            | Line radiation and Bremsstrahlung source; radiation losses are negative sources.
            particles_to_wall                          | 303            | Particle pumping by the wall; negative source for plasma and positive source for the wall
            particles_to_pump                          | 304            | Particle pumping by external pump; negative source for plasma and positive source for the pump
            charge_exchange                            | 305            | Source from charge exchange. Charge exchange losses are negative sources
            transport                                  | 400            | Source term related to transport processes
            neoclassical                               | 401            | Source term related to neoclassical processes
            equipartition                              | 402            | Equipartition due to collisions and turbulence
            turbulent_equipartition                    | 403            | Turbulent equipartition
            runaways                                   | 501            | Source from run-away processes; includes both electron and ion run-away
            ionisation                                 | 601            | Source from ionisation processes (not accounting for charge exchange)
            recombination                              | 602            | Source from recombination processes (not accounting for charge exchange)
            excitation                                 | 603            | Source from excitation processes
            database                                   | 801            | Source from database entry
            gaussian                                   | 802            | Artificial source with a gaussian profile
            custom_1                                   | 901            | Custom source terms 1; content to be decided by data provided
            custom_2                                   | 902            | Custom source terms 2; content to be decided by data provided
            custom_3                                   | 903            | Custom source terms 3; content to be decided by data provided
            custom_4                                   | 904            | Custom source terms 4; content to be decided by data provided
            custom_5                                   | 905            | Custom source terms 5; content to be decided by data provided
            custom_6                                   | 906            | Custom source terms 6; content to be decided by data provided
            custom_7                                   | 907            | Custom source terms 7; content to be decided by data provided
            custom_8                                   | 908            | Custom source terms 8; content to be decided by data provided
            custom_9                                   | 909            | Custom source terms 9; content to be decided by data provided
        """
        return Identifier(**self["identifier"]._as_dict())

    @sp_property
    def species(self):
        return CoreSourcesSpecies(self["species"], parent=self)

    @sp_property
    def global_quantities(self) -> CoreSourcesGlobalQuantities:
        return CoreSourcesGlobalQuantities(self["global_quantities"], parent=self)

    @sp_property
    def profiles_1d(self) -> CoreSourcesProfiles1D:
        return CoreSourcesProfiles1D(self["profiles_1d"], axis=self._grid.rho_tor_norm, parent=self)

    def update(self, *args, **kwargs) -> float:
        return super().update(*args, **kwargs)
        # res = self.profiles_1d.update(*args, **kwargs)
        # res += self.global_quantities.update(*args, **kwargs)


class CoreSources(IDS):
    """CoreSources
    """
    _IDS = "core_sources"
    Source = CoreSourcesSource

    def __init__(self, *args, grid: Optional[RadialGrid] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._grid = grid

    @sp_property
    def vacuum_toroidal_field(self) -> VacuumToroidalField:
        return VacuumToroidalField(**self["vacuum_toroidal_field"]._as_dict())

    @sp_property
    def source(self) -> List[CoreSourcesSource]:
        return List[CoreSourcesSource](self["source"], grid=self._grid, parent=self)

    def update(self,  *args,  **kwargs) -> float:
        redisual = sum([source.update(*args,  **kwargs) for source in self.source])
        return super().update(*args, **kwargs) + redisual
