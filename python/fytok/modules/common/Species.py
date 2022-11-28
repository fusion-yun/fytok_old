
import numpy as np
from scipy import constants
from spdm.logger import logger
from spdm.tags import _not_found_
from spdm.data import Dict, List, Node, sp_property

from ..constants.Atoms import atoms


class SpeciesElement(Dict):
    a: float = sp_property()
    """Mass of atom {dynamic} [Atomic Mass Unit] """

    atoms_n: int = sp_property()
    """Number of atoms of this element in the molecule {dynamic}"""

    z_n: int = sp_property()


class Species(Dict[Node]):
    Element = SpeciesElement

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        label = self.get("label", _not_found_)

        if label is _not_found_:
            raise RuntimeError(f"The label of species is not defined!")

        d_atom = atoms.get(label, _not_found_)
        if d_atom is _not_found_:
            raise RuntimeError(f"The data of species [{label}] is not defined!")
        else:
            self.update(d_atom)

    label: str = sp_property()
    """String identifying ion (e.g. H+, D+, T+, He+2, C+, ...) {dynamic}    """

    multiple_states_flag: int = sp_property(default=0)

    element: List[Element] = sp_property()

    @sp_property
    def a(self) -> float:
        """Mass of ion {dynamic} [Atomic Mass Unit]"""
        res = self.get("a", None)
        if res is None:
            res = sum([a.a*a.atoms_n for a in self.element])
        return res

    @sp_property
    def z(self) -> float:
        res = self.get("z", None)
        if res is None:
            res = self.get("z_ion", NotImplemented)
        return res


class SpeciesElectron(Species):

    @property
    def label(self) -> str:
        return "electron"

    @sp_property
    def a(self) -> float:
        """Mass of electron {dynamic} [Atomic Mass Unit]"""
        return constants.m_e/constants.m_u

    @sp_property
    def z(self) -> float:
        return -1


class SpeciesIonState(Dict):
    z_min: float = sp_property()
    """Minimum Z of the charge state bundle {dynamic} [Elementary Charge Unit]	FLT_0D	"""

    z_max: float = sp_property()
    """Maximum Z of the charge state bundle {dynamic} [Elementary Charge Unit]	FLT_0D	"""

    label: str = sp_property()
    """String identifying charge state (e.g. C+, C+2 , C+3, C+4, C+5, C+6, ...) {dynamic}	STR_0D	"""

    vibrational_level: float = sp_property()
    """Vibrational level (can be bundled) {dynamic} [Elementary Charge Unit]	FLT_0D	"""

    vibrational_mode: str = sp_property()
    """Vibrational mode of this state, e.g. "A_g". Need to define, or adopt a standard nomenclature. {dynamic}	STR_0D	"""

    electron_configuration: str = sp_property()
    """Configuration of atomic orbitals of this state, e.g. 1s2-2s1 {dynamic}	STR_0D	"""


class SpeciesIon(Species):

    is_impurity: bool = sp_property(default=False)

    has_fast_particle: bool = sp_property(default=False)

    z_ion: float = sp_property()
    """Ion charge (of the dominant ionisation state; lumped ions are allowed),
    volume averaged over plasma radius {dynamic} [Elementary Charge Unit]  FLT_0D  """

    neutral_index: int = sp_property(default=0)
    """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """

    z_ion_1d: np.ndarray = sp_property()
    """Average charge of the ion species (sum of states charge weighted by state density and
    divided by ion density) {dynamic} [-]  """

    z_ion_square_1d: np.ndarray = sp_property()
    """Average square charge of the ion species (sum of states square charge weighted by
      state density and divided by ion density) {dynamic} [-]  """

    multiple_states_flag: int = sp_property(default=0)
    """Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the state  {dynamic}    """

    state: List[SpeciesIonState] = sp_property()
    """Quantities related to the different states of the species (ionisation, energy, excitation, ...)  struct_array [max_size=unbounded]  1- 1...N"""
