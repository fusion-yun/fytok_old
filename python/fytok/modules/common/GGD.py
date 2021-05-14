from functools import cached_property
from spdm.data.Node import Dict
from spdm.util.logger import logger


class GGD(Dict):
    r"""General Grid Define
    """

    def __init__(self, cache, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def identifier(self):
        return NotImplemented

    class Space(Dict):
        def __init__(self, cache, *args, **kwargs):
            super().__init(*args, **kwargs)

    @cached_property
    def space(self):
        """Set of grid spaces"""
        return Dict(default_factory_array=lambda _holder=self: Mesh.Space(None, parent=_holder))

    class GridSubset(Dict):
        def __init__(self, cache, *args, **kwargs):
            super().__init(*args, **kwargs)

    @cached_property
    def grid_subset(self):
        """Grid subsets"""
        return Dict(default_factory_array=lambda _holder=self: Mesh.Space(None, parent=_holder))
