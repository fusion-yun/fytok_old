
import numpy as np
from enum import Enum

from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.Dict import Dict
from spdm.data.sp_property import sp_property

from ..common.IDS import IDS
from ..common.Misc import RZTuple, VacuumToroidalField, FloatArrya1D, Identifier
from ..common.Module import Module
from ..common.TimeSeries import TimeSeries, TimeSlice

class DistributionsProcess(Module):

    class Type(Enum):
        """
        Process type. 
            index=1 for NBI; 
            index=2 for nuclear reaction (reaction unspecified); 
            index=3 for nuclear reaction: T(d,n)4He [D+T->He4+n]; 
            index=4 for nuclear reaction: He3(d,p)4He [He3+D->He4+p]; 
            index=5 for nuclear reaction: D(d,p)T [D+D->T+p]; 
            index=6 for nuclear reaction: D(d,n)3He [D+D->He3+n]; 
            index=7 for runaway processes. 
            Available options (refer to the children of this identifier structure) :

        """
        unspecified = 0
        """unspecified"""
        NBI = 1
        """Source from neutral beam injection"""
        nuclear = 100
        """Source from nuclear reaction (reaction type unspecified)"""
        H_H_to_D_positron_nu = 101
        """Source from nuclear reaction: H+H->D+positron+neutrino"""
        H_D_to_He3_gamma = 102
        """Source from nuclear reaction: H+D->He3+gamma"""
        H_T_to_He3_n = 103
        """Source from nuclear reaction: H+T->He3+neutron"""
        H_He3_to_He4_positron_nu = 104
        """Source from nuclear reaction: H+He3->He4+positron+neutrino"""
        D_D_to_T_H = 105
        """Source from nuclear reaction: D+D->T+H"""
        D_D_to_He3_n = 106
        """Source from nuclear reaction: D+D->He3+neutron"""
        D_T_to_He4_n = 107
        """Source from nuclear reaction: T+D->He4+neutron"""
        D_He3_to_He4_H = 108
        """Source from nuclear reaction: He3+D->He4+H"""
        T_T_to_He4_n_n = 109
        """Source from nuclear reaction: T+T->He4+neutron+neutron"""
        T_He3_to_He4_H_n = 110
        """Source from nuclear reaction: He3+T->He4+H+neutron"""
        He3_He3_to_He4_H_H = 111
        """Source from nuclear reaction: He3+He3->He4+neutron+neutron"""
        He3_He4_to_Be7_gamma = 112
        """Source from nuclear reaction: He3+He4->Be7+gamma"""
        Li6_n_to_He4_T = 113
        """Source from nuclear reaction: Li6+n->He4+T"""
        Li7_n_to_He4_T_n = 114
        """Source from nuclear reaction: Li7+n->He4+T+n"""
        runaway = 1000
        """Source from runaway processes"""

    type: Type = sp_property()

    reactant_energy	: Identifier = sp_property()
    """For nuclear reaction source, energy of the reactants.
        index = 0 for a sum over all energies;
        index = 1 for thermal-thermal;
        index = 2 for beam-beam; index = 3 for beam-thermal """

    nbi_energy: Identifier = sp_property()
    """
    For NBI source, energy of the accelerated species considered.
        index = 0 for a sum over all energies;
        index = 1 for full energiy;
        index = 2 for half energy;
        index = 3 for third energy
    """
    nbi_unit: int = sp_property()
    """
    Index of the NBI unit considered. Refers to the "unit" array of the NBI IDS.
         0 means sum over all NBI units. {constant}
    """
    nbi_beamlets_group: int = sp_property()
    """
    Index of the NBI beamlets group considered. Refers to the "unit/beamlets_group" array of the NBI IDS.
    0 means sum over all beamlets groups. {constant}
    """


class DistributionsSource(TimeSlice):

    process: List[DistributionsProcess] = sp_property()

    gyro_type: int = sp_property()
    """Defines how to interpret the spatial coordinates: 
        1 = given at the actual particle birth point; 
        2 =given at the gyro centre of the birth point {constant}
    """
    class Species(Dict[Node]):
        """Species injected or consumed by this source/sink	"""
        class Type(Enum):
            """Species type. 
                index=1 for electron; 
                index=2 for ion species in a single/average state (refer to ion structure); 
                index=3 for ion species in a particular state (refer to ion/state structure); 
                index=4 for neutral species in a single/average state (refer to neutral structure); 
                index=5 for neutral species in a particular state (refer to neutral/state structure); 
                index=6 for neutron; index=7 for photon. 
                Available options (refer to the children of this identifier structure) 
            """
            unspecified = 0
            """unspecified"""
            electron = 1
            """Electron"""
            ion = 2
            """Ion species in a single/average state; refer to ion-structure"""
            ion_state = 3
            """Ion species in a particular state; refer to ion/state-structure"""
            neutral = 4
            """Neutral species in a single/average state; refer to neutral-structure"""
            neutral_state = 5
            """Neutral species in a particular state; refer to neutral/state-structure"""
            neutron = 6
            """Neutron"""
            photon = 7
            """Photon"""

        type: Type = sp_property()


class Distributions(IDS):
    r"""Distribution function(s) of one or many particle species. 
        This structure is specifically designed to handle non-Maxwellian distribution function 
        generated during heating and current drive, typically solved using a Fokker-Planck 
        calculation perturbed by a heating scheme(e.g. IC, EC, LH, NBI, or alpha heating) and 
        then relaxed by Coloumb collisions.   

        Note:
            Distributions is an ids
    """
    _IDS = "distributions"

    Source = DistributionsSource

    source: List[Source] = sp_property()

    vacuum_toroidal_field: VacuumToroidalField = sp_property()

    magnetic_axis: RZTuple[TimeSeries] = sp_property()

    time: FloatArrya1D = sp_property()
