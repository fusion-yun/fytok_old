import collections
from dataclasses import dataclass

from fytok.transport.MagneticCoordSystem import RadialGrid
from spdm.data.Node import Dict, List, Node, sp_property
from spdm.numlib import constants, np
from spdm.util.logger import logger


@dataclass
class Species:
    @dataclass
    class Element:
        a: float  # r"""Mass of atom {dynamic} [Atomic Mass Unit] """
        atoms_n: int  # r"""Number of atoms of this element in the molecule {dynamic}"""
        z_n: int

    label: str  # """String identifying ion (e.g. H+, D+, T+, He+2, C+, ...) {dynamic}    """

    multiple_states_flag: int

    element: List[Element]

    a: float      # """Mass of ion {dynamic} [Atomic Mass Unit]"""

    z: float


@dataclass
class SpeciesElectron(Species):
    label: str = "electron"
    a: float = constants.m_e/constants.m_u  # """Mass of elctron {dynamic} [Atomic Mass Unit]"""
    z: float = -1


@dataclass
class SpeciesIon(Species):

    @dataclass
    class State:

        z_min: int  # """Minimum Z of the charge state bundle {dynamic} [Elementary Charge Unit]	FLT_0D	"""

        z_max: int  # """Maximum Z of the charge state bundle {dynamic} [Elementary Charge Unit]	FLT_0D	"""

        label: str  # """String identifying charge state (e.g. C+, C+2 , C+3, C+4, C+5, C+6, ...) {dynamic}	STR_0D	"""

        vibrational_level: float  # """Vibrational level (can be bundled) {dynamic} [Elementary Charge Unit]	FLT_0D	"""

        vibrational_mode: str  # """Vibrational mode of this state, e.g. "A_g". Need to define, or adopt a standard nomenclature. {dynamic}	STR_0D	"""

        electron_configuration: str  # """Configuration of atomic orbitals of this state, e.g. 1s2-2s1 {dynamic}	STR_0D	"""

    z: float

    z_ion: float  # """Ion charge (of the dominant ionisation state; lumped ions are allowed),  volume averaged over plasma radius {dynamic} [Elementary Charge Unit]  FLT_0D  """

    neutral_index: int  # """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """

    z_ion_1d: float  # """Average charge of the ion species (sum of states charge weighted by state density and divided by ion density) {dynamic} [-]  """

    z_ion_square_1d: np.ndarray  # """Average square charge of the ion species (sum of states square charge weighted by  state density and divided by ion density) {dynamic} [-]  """

    multiple_states_flag: int  # """Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the state  {dynamic}    """

    state: List[State]  # """Quantities related to the different states of the species (ionisation, energy, excitation, ...)  struct_array [max_size=unbounded]  1- 1...N"""
