from functools import cached_property
from spdm.data.PhysicalGraph import PhysicalGraph, _next_
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module


class GGD(PhysicalGraph):
    r"""General Grid Define
    """

    def __init__(self, cache, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def identifier(self):
        return NotImplemented

    class Space(PhysicalGraph):
        def __init__(self, cache, *args, **kwargs):
            super().__init(*args, **kwargs)

    @cached_property
    def space(self):
        """Set of grid spaces"""
        return PhysicalGraph(default_factory_array=lambda _holder=self: Mesh.Space(None, parent=_holder))

    class GridSubset(PhysicalGraph):
        def __init__(self, cache, *args, **kwargs):
            super().__init(*args, **kwargs)

    @cached_property
    def grid_subset(self):
        """Grid subsets"""
        return PhysicalGraph(default_factory_array=lambda _holder=self: Mesh.Space(None, parent=_holder))
