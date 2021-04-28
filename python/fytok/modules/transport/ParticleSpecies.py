import collections
from functools import cached_property

import numpy as np
from spdm.data.AttributeTree import AttributeTree
from spdm.data.Node import List
from spdm.data.Profiles import Profiles
from spdm.util.logger import logger


class Species(Profiles):
    def __init__(self,   *args,  **kwargs):
        super().__init__(*args,   **kwargs)

    @cached_property
    def label(self):
        """String identifying ion (e.g. H+, D+, T+, He+2, C+, ...) {dynamic}    """
        return self.__raw_get__("label")

    class Element(AttributeTree):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

    @property
    def element(self):
        return List(self.__raw_get__("element"), default_factory=Species.Element, parent=self)

    class State(AttributeTree):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

    @property
    def multiple_states_flag(self):
        return self.__raw_get__("multiple_states_flag")

    @property
    def state(self):
        return List(self.__raw_get__("state"), default_factory=Species.State, parent=self)
