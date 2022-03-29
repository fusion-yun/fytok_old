from spdm.logger import logger
from spdm.data import Dict, File, Link, List, Node, Path, Query, sp_property,Function

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

    @sp_property
    def identifier(self):
        return Identifier(**self["identifier"]._as_dict())

    @sp_property
    def space(self) -> List[GGDSpace]:
        """Set of grid spaces"""
        return List[GGDSpace](self["space"], parent=self)

    @sp_property
    def grid_subset(self) -> List[GGDGridSubset]:
        """Grid subsets"""
        return List[GGDGridSubset](self["grid_subset"], parent=self)


class GGD(Dict):
    r"""General Grid Define
    """

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def grid(self) -> List[GGDGrid]:
        return List[GGDGrid](self["grid"], parent=self)

    def plot(self, axis=None, *args,  ggd=False, **kwargs):
        raise NotImplementedError()
