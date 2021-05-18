import collections
from functools import cached_property
import scipy.constants
import numpy as np
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Node import List, Dict
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


class SpeciesState(Dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class Species(Dict):
    Element = SpeciesElement
    State = SpeciesState

    def __init__(self,   *args,  **kwargs):
        super().__init__(*args,   **kwargs)

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

    @property
    def state(self) -> SpeciesState:
        return SpeciesState(self["state"],  parent=self)


class SpeciesElectron(Species):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


class SpeciesIon(Species):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def z(self) -> float:
        return self.z_ion

    @cached_property
    def z_ion(self) -> float:
        """Ion charge (of the dominant ionisation state; lumped ions are allowed),
        volume averaged over plasma radius {dynamic} [Elementary Charge Unit]  FLT_0D  """
        return self["z_ion"] or 1

    @cached_property
    def neutral_index(self) -> int:
        """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """
        return self["neutral_index"]

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
