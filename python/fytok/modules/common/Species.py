import collections
from functools import cached_property

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
        return self["a"]

    @cached_property
    def atoms_n(self):
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

    @property
    def element(self) -> List[Element]:
        return List[SpeciesElement](self["element"],  parent=self)

    @property
    def state(self) -> SpeciesState:
        return SpeciesState(self["state"],  parent=self)


class SpeciesIon(Species):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def z_ion(self) -> int:
        """Ion charge (of the dominant ionisation state; lumped ions are allowed),
        volume averaged over plasma radius {dynamic} [Elementary Charge Unit]  FLT_0D  """
        return self["z_ion"] or -1

    @cached_property
    def neutral_index(self):
        """Index of the corresponding neutral species in the ../../neutral array {dynamic}    """
        return self["neutral_index"] or None
