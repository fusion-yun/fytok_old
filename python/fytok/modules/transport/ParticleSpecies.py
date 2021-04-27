import collections
from functools import cached_property

import numpy as np
from spdm.data.List import List
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Function import Function
from spdm.util.logger import logger
from ...Profiles import Profiles


class Species(Profiles):
    def __init__(self,   *args,  **kwargs):
        super().__init__(*args,   **kwargs)

    @cached_property
    def label(self):
        """String identifying ion (e.g. H+, D+, T+, He+2, C+, ...) {dynamic}    """
        return self["label"]

    class Element(AttributeTree):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

    @property
    def element(self):
        return List(self._data["element"], default_factory=Species.Element, parent=self)

    class State:
        pass

    @property
    def multiple_states_flag(self):
        return 0

    @property
    def state(self):
        return List(self["state"], default_factory=Species.State, parent=self)
