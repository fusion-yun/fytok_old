import collections
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.constants
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Node import Dict, List
from spdm.data.Profiles import Profiles
from spdm.util.logger import logger


class SpeciesElement(Dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @cached_property
    def a(self):
        r"""Mass of atom {dynamic} [Atomic Mass Unit] """
        return self["a"]

    @cached_property
    def atoms_n(self):
        r"""Number of atoms of this element in the molecule {dynamic}"""
        return self["atoms_n"]

    @cached_property
    def z_n(self):
        return self["z_n"]


class Species(Profiles):
    Element = SpeciesElement

    def __init__(self,   *args, axis=None, parent=None,  **kwargs):
        super().__init__(*args, axis=axis if axis is not None else parent.grid.rho_tor_norm, parent=parent, **kwargs)

    @cached_property
    def label(self) -> str:
        """String identifying ion (e.g. H+, D+, T+, He+2, C+, ...) {dynamic}    """
        return self["label"]

    @cached_property
    def multiple_states_flag(self) -> int:
        return self["multiple_states_flag"]

    @cached_property
    def element(self) -> List[Element]:
        return List[SpeciesElement](self["element"],  parent=self)

    @cached_property
    def a(self):
        """Mass of ion {dynamic} [Atomic Mass Unit]"""
        return np.sum([a.a*a.atoms_n for a in self.element])

    @cached_property
    def z(self) -> float:
        return NotImplemented

  

class SpeciesElectron(Species):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,  **kwargs)

    @property
    def label(self) -> str:
        return "electron"

    @cached_property
    def a(self):
        """Mass of elctron {dynamic} [Atomic Mass Unit]"""
        return scipy.constants.m_e/scipy.constants.m_u

    @cached_property
    def z(self) -> float:
        return -1


class SpeciesIonState(Dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @cached_property
    def z_min(self):
        """Minimum Z of the charge state bundle {dynamic} [Elementary Charge Unit]	FLT_0D	"""
        return self["z_min"]

    @cached_property
    def z_max(self):
        """Maximum Z of the charge state bundle {dynamic} [Elementary Charge Unit]	FLT_0D	"""
        return self["z_max"]

    @cached_property
    def label(self):
        """String identifying charge state (e.g. C+, C+2 , C+3, C+4, C+5, C+6, ...) {dynamic}	STR_0D	"""
        return self["label"]

    @cached_property
    def vibrational_level(self):
        """Vibrational level (can be bundled) {dynamic} [Elementary Charge Unit]	FLT_0D	"""
        return self["vibrational_level"]

    @cached_property
    def vibrational_mode(self):
        """Vibrational mode of this state, e.g. "A_g". Need to define, or adopt a standard nomenclature. {dynamic}	STR_0D	"""
        return self["vibrational_mode"]

    @cached_property
    def electron_configuration(self):
        """Configuration of atomic orbitals of this state, e.g. 1s2-2s1 {dynamic}	STR_0D	"""
        return self["electron_configuration"]


class SpeciesIon(Species):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args,   **kwargs)

    @cached_property
    def z(self) -> float:
        return self.z_ion

    @cached_property
    def z_ion(self) -> float:
        """Ion charge (of the dominant ionisation state; lumped ions are allowed),
        volume averaged over plasma radius {dynamic} [Elementary Charge Unit]  FLT_0D  """
        return self._entry.get("z_ion") or 1

    @cached_property
    def neutral_index(self) -> int:
        """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """
        return self._entry.get("neutral_index")

    @cached_property
    def z_ion_1d(self):
        """Average charge of the ion species (sum of states charge weighted by state density and
        divided by ion density) {dynamic} [-]  """
        return NotImplemented

    @cached_property
    def z_ion_square_1d(self):
        """Average square charge of the ion species (sum of states square charge weighted by
        state density and divided by ion density) {dynamic} [-]  """
        return NotImplemented

    @cached_property
    def multiple_states_flag(self) -> int:
        """Multiple states calculation flag : 0-Only one state is considered; 1-Multiple states are considered and are described in the state  {dynamic}    """
        return self._entry.get("multiple_states_flag") or 0

    @cached_property
    def state(self):
        """Quantities related to the different states of the species (ionisation, energy, excitation, ...)  struct_array [max_size=unbounded]  1- 1...N"""
        return List[SpeciesIonState](self["state"], parent=self)
