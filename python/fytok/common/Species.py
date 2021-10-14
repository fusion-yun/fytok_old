import collections
from dataclasses import dataclass

from fytok.common.Atoms import atoms
from fytok.transport.MagneticCoordSystem import RadialGrid
from scipy import constants
from spdm.data.Node import Dict, List, Node, sp_property
from spdm.util.logger import logger


class SpeciesElement(Dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @sp_property
    def a(self) -> float:
        r"""Mass of atom {dynamic} [Atomic Mass Unit] """
        return self["a"]

    @sp_property
    def atoms_n(self) -> int:
        r"""Number of atoms of this element in the molecule {dynamic}"""
        return self["atoms_n"]

    @sp_property
    def z_n(self) -> int:
        return self["z_n"]


class Species(Dict[Node]):
    Element = SpeciesElement

    def __init__(self,  d, /, **kwargs):
        super().__init__(d, ** collections.ChainMap(kwargs, atoms.get(d.get("label"), {})))

    @property
    def nid(self) -> str:
        return self.get("label", None)

    @sp_property
    def label(self) -> str:
        """String identifying ion (e.g. H+, D+, T+, He+2, C+, ...) {dynamic}    """
        return self.get("label", "")

    @sp_property
    def multiple_states_flag(self) -> int:
        return self.get("multiple_states_flag", 0)

    @sp_property
    def element(self) -> List[Element]:
        return self.get("element", [])

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
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)

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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @sp_property
    def z_min(self):
        """Minimum Z of the charge state bundle {dynamic} [Elementary Charge Unit]	FLT_0D	"""
        return self["z_min"]

    @sp_property
    def z_max(self):
        """Maximum Z of the charge state bundle {dynamic} [Elementary Charge Unit]	FLT_0D	"""
        return self["z_max"]

    @sp_property
    def label(self):
        """String identifying charge state (e.g. C+, C+2 , C+3, C+4, C+5, C+6, ...) {dynamic}	STR_0D	"""
        return self["label"]

    @sp_property
    def vibrational_level(self):
        """Vibrational level (can be bundled) {dynamic} [Elementary Charge Unit]	FLT_0D	"""
        return self["vibrational_level"]

    @sp_property
    def vibrational_mode(self):
        """Vibrational mode of this state, e.g. "A_g". Need to define, or adopt a standard nomenclature. {dynamic}	STR_0D	"""
        return self["vibrational_mode"]

    @sp_property
    def electron_configuration(self):
        """Configuration of atomic orbitals of this state, e.g. 1s2-2s1 {dynamic}	STR_0D	"""
        return self["electron_configuration"]


class SpeciesIon(Species):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args,  **kwargs)

    @sp_property
    def is_impurity(self) -> bool:
        return self.get("is_impurity", False)

    @sp_property
    def has_fast_particle(self) -> bool:
        return self.get("has_fast_particle", False)

    @sp_property
    def z_ion(self) -> float:
        """Ion charge (of the dominant ionisation state; lumped ions are allowed),
        volume averaged over plasma radius {dynamic} [Elementary Charge Unit]  FLT_0D  """
        return self.z

    @sp_property
    def neutral_index(self) -> int:
        """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """
        return self.get("neutral_index", 0)

    @sp_property
    def z_ion_1d(self):
        """Average charge of the ion species (sum of states charge weighted by state density and
        divided by ion density) {dynamic} [-]  """
        return NotImplemented

    @sp_property
    def z_ion_square_1d(self):
        """Average square charge of the ion species (sum of states square charge weighted by
        state density and divided by ion density) {dynamic} [-]  """
        return NotImplemented

    @sp_property
    def multiple_states_flag(self) -> int:
        """Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the state  {dynamic}    """
        return self.get("multiple_states_flag", 0)

    @sp_property
    def state(self) -> List[SpeciesIonState]:
        """Quantities related to the different states of the species (ionisation, energy, excitation, ...)  struct_array [max_size=unbounded]  1- 1...N"""
        return self.get("state")
