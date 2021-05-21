from functools import cached_property

from spdm.data.Node import Dict, List
from spdm.flow.Actor import Actor
from spdm.util.logger import logger

from ..common.Misc import Identifier


class GGDGridSubset(Dict):
    def __init__(self,  *args, **kwargs):
        super().__init(*args, **kwargs)


class GGDSpace(Dict):
    def __init__(self, cache, *args, **kwargs):
        super().__init(*args, **kwargs)


class GGDGrid(Dict):
    def __init__(self,   *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def identifier(self):
        return Identifier(**self["identifier"]._as_dict())

    @cached_property
    def space(self) -> List[GGDSpace]:
        """Set of grid spaces"""
        return List[GGDSpace](self["space"], parent=self)

    @cached_property
    def grid_subset(self) -> List[GGDGridSubset]:
        """Grid subsets"""
        return List[GGDGridSubset](self["grid_subset"], parent=self)


class GGD(Actor):
    r"""General Grid Define
    """

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def grid(self) -> List[GGDGrid]:
        return List[GGDGrid](self["grid"], parent=self)

    def plot(self, axis=None, *args,  ggd=False, **kwargs):
        raise NotImplementedError()
